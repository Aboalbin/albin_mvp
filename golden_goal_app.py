# golden_goal_app.py
# Golden Goal – interaktiv karta + förslag på sponsorer
# Läser data från ./data/ (stöd för .csv och .xlsx), geokodar förening vid behov,
# och föreslår närliggande, storleksliknande företag. Storleksmatchning påverkar
# listan tydligt (strikt filter + tung vikt i rankingen).

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------- Hjälpare (koordinater) ----------

# --- Koordinat-hjälp (ersätt hela blocket från SWEDEN_BBOX till enrich_coords_for_map) ---
SWEDEN_BBOX = (55.0, 10.0, 69.5, 25.5)  # lat_min, lon_min, lat_max, lon_max

def _in_sweden(lat, lon):
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    return lat.between(SWEDEN_BBOX[0], SWEDEN_BBOX[2]) & lon.between(SWEDEN_BBOX[1], SWEDEN_BBOX[3])

def _maybe_fix_coordinates_inplace(df: pd.DataFrame, label: str = "companies"):
    """
    Säker fix per rad:
      1) Byt plats på rader där (lat≈10–25 & lon≈55–70) → uppenbart omkastat.
      2) Konvertera ENDAST rader som ser ut som meter (SWEREF 99 TM, EPSG:3006) till WGS84.
      3) Värden utanför Sverige → sätt NaN (så kan vi geokoda just de raderna).
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # 1) Byt plats på uppenbart omkastade rader
    mask_swap = df["latitude"].between(10, 25) & df["longitude"].between(55, 70)
    if mask_swap.any():
        tmp = df.loc[mask_swap, "latitude"].copy()
        df.loc[mask_swap, "latitude"] = df.loc[mask_swap, "longitude"]
        df.loc[mask_swap, "longitude"] = tmp

    # 2) Konvertera endast rader som ser ut att vara i meter
    mask_m = (df["latitude"].abs() > 90) | (df["longitude"].abs() > 180)
    if mask_m.any():
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(3006, 4326, always_xy=True)  # (E,N) -> (lon,lat)
            x = df.loc[mask_m, "longitude"].astype(float).values  # Easting
            y = df.loc[mask_m, "latitude"].astype(float).values   # Northing
            lon2, lat2 = transformer.transform(x, y)
            df.loc[mask_m, "latitude"] = lat2
            df.loc[mask_m, "longitude"] = lon2
        except Exception:
            pass

    # 3) Allt som fortfarande hamnar utanför Sverige → NaN (så geokodar vi just dem)
    bad = ~_in_sweden(df["latitude"], df["longitude"])
    if bad.any():
        df.loc[bad, ["latitude", "longitude"]] = np.nan

@st.cache_data(show_spinner=False)
def geocode_company(name: str, district: str | None = None):
    """Geokodar enstaka bolag för kartan när koordinater saknas/är orimliga."""
    try:
        g = Nominatim(user_agent="golden_goal_app/companies", timeout=8)
        q = ", ".join([p for p in [name, district, "Sweden"] if p])
        loc = g.geocode(q, exactly_one=True, country_codes="se", addressdetails=False)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception:
        pass
    return None, None

def enrich_coords_for_map(df_results: pd.DataFrame, max_fix: int = 15) -> pd.DataFrame:
    """
    För kartan: försök automatiskt rätta typiska fel (swap/meter) och geokoda
    HÖGST 'max_fix' rader som saknar koordinater efter fix.
    """
    df = df_results.copy()
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    _maybe_fix_coordinates_inplace(df)

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    bad = lat.isna() | lon.isna()

    # Geokoda bara de rader som saknar koordinater efter fixen
    for idx in list(df.index[bad])[:max_fix]:
        name = str(df.at[idx, "name"])
        district = str(df.at[idx, "district"]) if "district" in df.columns and pd.notna(df.at[idx, "district"]) else None
        glat, glon = geocode_company(name, district)
        if glat is not None and glon is not None:
            df.at[idx, "latitude"] = glat
            df.at[idx, "longitude"] = glon

    return df


# ---------- Hjälpare (text & parsing) ----------

def _clean_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())

def _norm_text(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    return str(s).strip().lower()

# ---------- Normalisering: Föreningar ----------

def normalize_association_columns(df: pd.DataFrame):
    norm = {_clean_col(c): c for c in df.columns}
    colmap = {}
    name_alias = ["name","namn","forening","förening","association","club","klubb","org","organisation"]
    lat_alias  = ["latitude","lat","latitud","breddgrad","northing","y"]
    lon_alias  = ["longitude","lon","long","longitud","längdgrad","easting","x"]
    address_alias = ["address","gatuadress","street","adress"]
    postal_alias  = ["postal_code","postnummer","postkod"]
    city_alias    = ["city","postort","ort","stad"]
    muni_alias    = ["municipality","kommun"]

    def pick(aliases):
        for a in aliases:
            if a in norm: return norm[a]
        return None

    colmap["name"]         = pick(name_alias)
    colmap["latitude"]     = pick(lat_alias)
    colmap["longitude"]    = pick(lon_alias)
    colmap["address"]      = pick(address_alias)
    colmap["postal_code"]  = pick(postal_alias)
    colmap["city"]         = pick(city_alias)
    colmap["municipality"] = pick(muni_alias)

    out = df.copy()
    for std in ["name","latitude","longitude","address","postal_code","city","municipality"]:
        src = colmap.get(std)
        if src and src != std:
            out = out.rename(columns={src: std})

    warnings = []
    need = [c for c in ["name","latitude","longitude"] if c not in out.columns]
    if need:
        warnings.append("Föreningsfilen saknar tydliga kolumner för: " + ", ".join(need) + ".")

    if "latitude" in out.columns:
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    if "longitude" in out.columns:
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    return out, warnings

# ---------- Normalisering: Företag ----------

def derive_size_bucket_from_employees(val):
    try:
        n = float(str(val).replace(" ", "").replace(",", "."))
    except Exception:
        return None
    if pd.isna(n): return None
    if n < 10:  return "small"
    if n < 50:  return "medium"
    return "large"

def bin4_from_employees(val):
    try:
        n = float(str(val).replace(" ", "").replace(",", "."))
    except Exception:
        return np.nan
    if pd.isna(n): return np.nan
    if n < 250:  return 0
    if n < 500:  return 1
    if n < 1000: return 2
    return 3

def _first_textual_column(df: pd.DataFrame, exclude: set):
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            return c
    return None

def _parse_size_bin4(val):
    """Returnerar 0..3 för text/numerik som '0-249', '250–499', '500-999', '1000+' eller ett tal."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return np.nan
    s = str(val).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("–", "-").replace("—", "-")

    if s.endswith("+"):
        try:
            n = float(s[:-1].replace(",", "."))
            return 3 if n >= 1000 else (2 if n >= 500 else (1 if n >= 250 else 0))
        except Exception:
            return np.nan
    m = re.match(r"^(\d+)-(\d+)$", s)
    if m:
        hi = max(float(m.group(1)), float(m.group(2)))
        if hi < 250:   return 0
        if hi < 500:   return 1
        if hi < 1000:  return 2
        return 3
    try:
        n = float(s.replace(",", "."))
        if n < 250:   return 0
        if n < 500:   return 1
        if n < 1000:  return 2
        return 3
    except Exception:
        return np.nan

def normalize_company_columns(df: pd.DataFrame):
    norm = {_clean_col(c): c for c in df.columns}
    colmap = {}

    name_alias = ["name","namn","company","company_name","foretagsnamn","företagsnamn","bolagsnamn","bolag","organisation"]
    lat_alias  = ["latitude","lat","latitud","y","breddgrad"]
    lon_alias  = ["longitude","lon","long","longitud","x","lng","längdgrad"]
    size_alias = ["size_bucket","size","storlek","storlekskategori","employee_bucket","employee_range"]
    employees_alias = ["employees","employee_count","anstallda","anställda","antal_anstallda"]
    district_alias  = ["district","kommun","postort","ort","stad","city","municipality","lan","län","county","region"]

    def pick(aliases):
        for a in aliases:
            if a in norm: return norm[a]
        return None

    colmap["name"]        = pick(name_alias)
    colmap["latitude"]    = pick(lat_alias)
    colmap["longitude"]   = pick(lon_alias)
    colmap["size_bucket"] = pick(size_alias)
    colmap["employees"]   = pick(employees_alias)
    colmap["district"]    = pick(district_alias)

    out = df.copy()
    for std in ["name","latitude","longitude","district"]:
        src = colmap.get(std)
        if src and src != std:
            out = out.rename(columns={src: std})

    if "latitude" in out.columns:
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    if "longitude" in out.columns:
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    warnings = []
    if colmap["size_bucket"] is None:
        emp_col = colmap["employees"]
        if emp_col and emp_col in out.columns:
            out["size_bucket"] = out[emp_col].apply(derive_size_bucket_from_employees)
        else:
            warnings.append("Bolagsfilen saknar 'size_bucket'. Storlek vägs svagare.")
    else:
        if colmap["size_bucket"] != "size_bucket":
            out = out.rename(columns={colmap["size_bucket"]: "size_bucket"})
        out["size_bucket"] = out["size_bucket"].astype(str).str.strip().str.lower()

    if "name" not in out.columns or out["name"].isna().all():
        candidate = _first_textual_column(out, exclude={"latitude","longitude","size_bucket","district"})
        if candidate:
            out = out.rename(columns={candidate: "name"})
        else:
            out["name"] = [f"Okänt bolag #{i+1}" for i in range(len(out))]

    miss = [k for k in ["name","latitude","longitude"] if k not in out.columns]
    if miss:
        warnings.append("Bolagsfilen saknar kolumner för: " + ", ".join(miss) + ".")

    if colmap["district"] and colmap["district"] != "district":
        out = out.rename(columns={colmap["district"]: "district"})

    # 4-stegs bucket
    if "size_bin4" not in out.columns:
        out["size_bin4"] = np.nan

    if out["size_bin4"].isna().all() and colmap.get("employees") and colmap["employees"] in out.columns:
        out["size_bin4"] = out[colmap["employees"]].apply(bin4_from_employees)

    if out["size_bin4"].isna().all() and "size_bucket" in out.columns:
        out["size_bin4"] = out["size_bucket"].map({"small":0,"medium":1,"large":3})

    if out["size_bin4"].isna().all():
        candidate_cols = []
        for c in ["size_bucket", colmap.get("size_bucket"), colmap.get("employees"),
                  "employees", "anställda", "anstallda", "employee_range", "antal_anstallda",
                  "storlek", "size"]:
            if c and c in out.columns and c not in candidate_cols:
                candidate_cols.append(c)
        for c in candidate_cols:
            parsed = out[c].apply(_parse_size_bin4)
            if parsed.notna().any():
                out["size_bin4"] = parsed
                break

    if out["size_bin4"].isna().all():
        warnings.append("Kunde inte tolka företagsstorlek – förslag styrs mest av geografi.")

    return out, warnings

# ---------- Data-laddning ----------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _read_any(name: str) -> pd.DataFrame:
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    xlsx_path = os.path.join(DATA_DIR, f"{name}.xlsx")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    if os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path, engine="openpyxl")
    raise FileNotFoundError(f"Hittade varken {csv_path} eller {xlsx_path}")

@st.cache_data
def load_data():
    companies_raw = _read_any("companies_prepared")
    associations_raw = _read_any("associations_prepared")
    associations, assoc_warnings = normalize_association_columns(associations_raw)
    companies,    comp_warnings  = normalize_company_columns(companies_raw)
    _maybe_fix_coordinates_inplace(companies)
    return associations, companies, assoc_warnings, comp_warnings

# ---------- Geokodning av förening ----------

@st.cache_data(show_spinner=False)
def geocode_association(name: str, address: str|None, postal_code: str|None, city: str|None, municipality: str|None):
    """Nominatim-geokodning, flerstegssökning. Returnerar (lat, lon, label) eller (None, None, None)."""
    try:
        geolocator = Nominatim(user_agent="golden_goal_app/1.1", timeout=10)

        def _q(*parts):
            return ", ".join([str(p) for p in parts if p and str(p).strip()])

        # Rensa klubbnamn (suffix + genitiv-s)
        base = (name or "").strip()
        tokens = base.split()
        club_suffixes = {"if","ik","is","ff","bk","fk","gf","hf","hk","rf","cf","ifk","aif","sk","sf"}
        if tokens and tokens[-1].lower().strip(".,-()") in club_suffixes:
            tokens = tokens[:-1]
        base_clean = " ".join(tokens).rstrip("s").strip() or name

        queries = []
        parts = []
        if address: parts.append(address)
        if postal_code: parts.append(postal_code)
        if city: parts.append(city)
        if municipality and (not city or _norm_text(city) != _norm_text(municipality)):
            parts.append(municipality)
        parts.append("Sweden")
        if len(parts) <= 2:
            parts.insert(0, name)
        queries.append(_q(*parts))
        if municipality: queries.append(_q(name, municipality, "Sweden"))
        if city:         queries.append(_q(name, city, "Sweden"))
        if municipality: queries.append(_q(base_clean, municipality, "Sweden"))
        if city:         queries.append(_q(base_clean, city, "Sweden"))
        queries.append(_q(base_clean, "Sweden"))
        if municipality: queries.append(_q(municipality, "Sweden"))
        if city:         queries.append(_q(city, "Sweden"))

        best = None
        best_score = -1.0
        muni_norm = _norm_text(municipality)
        city_norm = _norm_text(city)
        name_norm = _norm_text(base_clean)

        for q in queries:
            if not q:
                continue
            results = geolocator.geocode(q, exactly_one=False, country_codes="se", addressdetails=True, limit=5)
            if not results:
                continue
            for loc in results:
                addr = getattr(loc, "raw", {}).get("address", {})
                muni = _norm_text(addr.get("municipality") or addr.get("town") or addr.get("city") or addr.get("village"))
                score = 0.0
                if muni_norm and muni and muni_norm == muni: score += 2.0
                if city_norm and muni and city_norm == muni: score += 1.0
                if name_norm and name_norm in _norm_text(loc.address): score += 0.5
                if score > best_score:
                    best, best_score = loc, score
            if best is not None and best_score >= 2.0:
                break
        if best is None:
            return None, None, None
        return float(best.latitude), float(best.longitude), best.address
    except Exception:
        return None, None, None

def resolve_assoc_coords(assoc_row: pd.Series):
    csv_lat = assoc_row.get("latitude")
    csv_lon = assoc_row.get("longitude")
    has_csv_coords = csv_lat is not None and not pd.isna(csv_lat) and csv_lon is not None and not pd.isna(csv_lon)

    name = assoc_row.get("name")
    address = assoc_row.get("address")
    postal  = assoc_row.get("postal_code")
    city    = assoc_row.get("city")
    muni    = assoc_row.get("municipality")

    if name and (address or city or muni):
        lat, lon, label = geocode_association(str(name), address, postal, city, muni)
        if lat is not None and lon is not None:
            return lat, lon, label
    if has_csv_coords:
        return float(csv_lat), float(csv_lon), None
    return None, None, None

# ---------- Geo & storlek – kandidatlogik ----------

def haversine_km(lat1, lon1, lat2, lon2):
    """Vektoriserad haversine (km). lat/lon i grader; lat2/lon2 kan vara arr/Series."""
    R = 6371.0088
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def _allowed_size_bins_strict(assoc_bin: int) -> list[int]:
    """Strikt storlekskarta — gör att valet påverkar listan tydligt."""
    mapping = {
        0: [0],        # 0–249 → små företag
        1: [0, 1],     # 250–499 → små & medel
        2: [2, 3],     # 500–999 → stora & mycket stora
        3: [3],        # 1000+ → mycket stora
    }
    return mapping.get(int(assoc_bin), [0, 1, 2, 3])

def match_sponsors(lat, lon, size_bin4, df_companies: pd.DataFrame, top_k=10):
    """
    Tvåstegslogik:
    1) Förfilter på avstånd (start 60 km, väx upp till 300 km vid behov).
    2) Strikt storleksfilter (allowed bins) – relaxa gradvis om få träffar.
    3) Rankning: 70 % storlek, 30 % närhet.
    """
    df = df_companies.copy()
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["size_bin4"] = pd.to_numeric(df.get("size_bin4", np.nan), errors="coerce")

    df = df.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    if len(df) == 0:
        return df

    # Avstånd till föreningen (vektoriserat)
    df["distance_km"] = haversine_km(lat, lon, df["latitude"].values, df["longitude"].values)

    # Geo-förfilter med adaptiv radie
    min_needed = max(top_k * 3, 30)
    for radius in [60, 120, 200, 300, np.inf]:
        pool = df[df["distance_km"] <= radius].copy()
        if len(pool) >= min_needed or radius == np.inf:
            break

    # Strikt storleksfilter, relaxa om få kandidater
    allowed = set(_allowed_size_bins_strict(size_bin4))
    pool_pref = pool[pool["size_bin4"].isin(list(allowed))].copy()

    if len(pool_pref) < top_k:            # relax 1: tillåt ±1 steg
        allowed_relax = set(allowed)
        allowed_relax.update([max(0, size_bin4-1), min(3, size_bin4+1)])
        pool_pref = pool[pool["size_bin4"].isin(list(allowed_relax))].copy()

    if len(pool_pref) < top_k:            # relax 2: tillåt alla
        pool_pref = pool.copy()

    # Poäng: storlek (0..1) + geo (0..1)
    size_gap = (pool_pref["size_bin4"] - float(size_bin4)).abs().clip(upper=3.0).fillna(3.0)
    pool_pref["size_score"] = 1.0 - (size_gap / 3.0)

    if pool_pref["distance_km"].max() == pool_pref["distance_km"].min():
        pool_pref["geo_score"] = 1.0
    else:
        pool_pref["geo_score"] = 1.0 - (pool_pref["distance_km"] - pool_pref["distance_km"].min()) / \
                                 (pool_pref["distance_km"].max() - pool_pref["distance_km"].min())

    W_SIZE, W_GEO = 0.70, 0.30
    pool_pref["score"] = W_SIZE * pool_pref["size_score"] + W_GEO * pool_pref["geo_score"]

    keep = ["name","district","latitude","longitude","distance_km","size_bin4","size_score","geo_score","score"]
    for c in keep:
        if c not in pool_pref.columns:
            pool_pref[c] = np.nan

    out = (pool_pref
           .sort_values(["score","size_score","geo_score"], ascending=False)
           .head(int(top_k))
           .reset_index(drop=True))[keep]

    return out

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Golden Goal", layout="wide")
st.title("🥇 Golden Goal")
st.markdown("Välj en förening och ange dess storlek. Appen föreslår företag baserat på **geografisk närhet** och **storlekslikhet** (storlek väger tungt).")

with st.sidebar:
    st.header("Inställningar")
    show_debug = st.checkbox("Visa debug-varningar", value=False)
    st.caption("Storleksvalet påverkar listan tydligt: 0–249 → små bolag, 1000+ → mycket stora, osv.")

associations, companies, assoc_warnings, comp_warnings = load_data()
if show_debug:
    if assoc_warnings:
        for w in assoc_warnings: st.warning("Föreningsdata: " + w)
    if comp_warnings:
        for w in comp_warnings:  st.warning("Bolagsdata: " + w)

required_columns_assoc = ['name']
missing_cols_assoc = [col for col in required_columns_assoc if col not in associations.columns]
if missing_cols_assoc:
    st.error(f"❌ Följande kolumner saknas i föreningsdatan: {', '.join(missing_cols_assoc)}")
    st.stop()

# Vi använder företagsdatan direkt (med fixade koordinater)
df_companies = companies.copy()
if len(df_companies) == 0:
    st.error("Inga företag i datan. Kontrollera ./data/companies_prepared.*")
    st.stop()

if show_debug:
    if "size_bin4" in df_companies:
        vc = df_companies["size_bin4"].value_counts(dropna=True).sort_index()
        st.caption("Företagsstorlek (bin4) i data: " + ", ".join([f"{int(k)}:{int(v)}" for k, v in vc.items()]))
    lat_rng = (df_companies["latitude"].min(), df_companies["latitude"].max())
    lon_rng = (df_companies["longitude"].min(), df_companies["longitude"].max())
    st.caption(f"Koordinatspann (företag): lat {lat_rng[0]:.3f}..{lat_rng[1]:.3f}, lon {lon_rng[0]:.3f}..{lon_rng[1]:.3f}")

left, right = st.columns([1.2, 1.8])

with left:
    # --- Välj förening (utan förvalt värde) ---
    assoc_names = associations['name'].dropna().astype(str).tolist()
    assoc_options = ["— Välj förening —"] + assoc_names
    if "gg_assoc_select" not in st.session_state:
        st.session_state["gg_assoc_select"] = "— Välj förening —"

    selected_label = st.selectbox("Välj förening", assoc_options, index=0, key="gg_assoc_select")
    selected_assoc = None if selected_label == "— Välj förening —" else selected_label

    SIZE_OPTIONS = ["— Välj storlek —", "0–249", "250–499", "500–999", "1000+"]
    size_choice = st.selectbox("Föreningens storlek (obligatorisk)", SIZE_OPTIONS, index=0, key="gg_size_choice")
    size_map = {"0–249": 0, "250–499": 1, "500–999": 2, "1000+": 3}
    top_k = st.number_input("Antal förslag", min_value=5, max_value=50, value=10, step=1, key="gg_topk")

    # Nollställ resultat när förening byts
    if st.session_state.get("gg_prev_assoc") != selected_assoc:
        st.session_state["gg_prev_assoc"] = selected_assoc
        st.session_state.pop("gg_results_display", None)
        st.session_state.pop("gg_results_raw", None)
        st.session_state.pop("gg_assoc_info", None)

    can_search = (selected_assoc is not None) and (size_choice in size_map)
    btn = st.button("Uppdatera förslag", disabled=not can_search)
    if not can_search:
        st.info("Välj förening och storlek för att kunna visa förslag.")

    if can_search and btn:
        assoc_row = associations[associations['name'] == selected_assoc].iloc[0]
        lat_res, lon_res, label = resolve_assoc_coords(assoc_row)

        if lat_res is None or lon_res is None:
            st.warning("Kunde inte hitta koordinater för föreningen. Ange plats manuellt nedan.")
            with st.expander("Ange plats manuellt"):
                col1, col2 = st.columns(2)
                with col1:
                    man_lat = st.number_input("Latitude (WGS84)", value=57.708870, step=0.0001, format="%.6f")
                with col2:
                    man_lon = st.number_input("Longitude (WGS84)", value=11.974560, step=0.0001, format="%.6f")
                use_manual = st.checkbox("Använd manuellt angiven plats", value=True)
            if use_manual:
                lat_res, lon_res = float(man_lat), float(man_lon)
            else:
                st.stop()

        st.write(f"**Koordinater (karta):** ({lat_res:.6f}, {lon_res:.6f})")
        if show_debug and label:
            st.caption(f"Adress från geokodning: {label}")

        assoc_size_bin4 = size_map[size_choice]
        results = match_sponsors(lat_res, lon_res, assoc_size_bin4, df_companies, top_k=int(top_k))

        if len(results) > 0:
            display = pd.DataFrame({
                "Företag": results["name"],
                "Score": results["score"].round(3)
            })
            st.session_state["gg_results_display"] = display
            st.session_state["gg_results_raw"] = results
            st.session_state["gg_assoc_info"] = {
                "name": selected_assoc,
                "lat": lat_res,
                "lon": lon_res,
                "size_bin4": assoc_size_bin4,
            }
        else:
            st.warning("Inga kandidater hittades – kontrollera att företagsdatan innehåller lat/long.")

    if st.session_state.get("gg_results_display") is not None:
        st.dataframe(st.session_state["gg_results_display"], use_container_width=True, hide_index=True)
        st.caption("Tabellen visar företagsnamn och score (storlek väger 70 %).")

with right:
    st.subheader("Karta")

    raw = st.session_state.get("gg_results_raw", None)
    assoc_info = st.session_state.get("gg_assoc_info", None)

    m = folium.Map(location=[62.0, 15.0], zoom_start=5, tiles="OpenStreetMap")
    points = []

    if assoc_info is not None:
        folium.Marker(
            [assoc_info["lat"], assoc_info["lon"]],
            popup=folium.Popup(f"<b>{assoc_info['name']}</b><br>(Förening)"),
            tooltip=f"Förening: {assoc_info['name']}",
            icon=folium.Icon(color="blue", icon="star")
        ).add_to(m)
        points.append([assoc_info["lat"], assoc_info["lon"]])

    if raw is not None and len(raw) > 0:
        try:
            map_df = enrich_coords_for_map(raw, max_fix=15)
        except NameError:
            map_df = raw

        cluster = MarkerCluster().add_to(m)
        for _, r in map_df.iterrows():
            lat = pd.to_numeric(r.get("latitude"), errors="coerce")
            lon = pd.to_numeric(r.get("longitude"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue

            name = str(r.get("name", "Okänt företag"))
            district = r.get("district")
            district = str(district) if pd.notna(district) else "okänd"
            dist = r.get("distance_km")
            dist_txt = f"{dist:.1f} km" if (dist is not None and not pd.isna(dist)) else "–"
            score_val = r.get("score", np.nan)
            score_txt = f"{float(score_val):.3f}" if pd.notna(score_val) else "–"

            html = f"""
            <div style='font-size:14px'>
                <b>{name}</b><br>
                Ort: {district}<br>
                Avstånd: {dist_txt}<br>
                Score: {score_txt}
            </div>
            """
            folium.Marker(
                [float(lat), float(lon)],
                popup=folium.Popup(html, max_width=300),
                tooltip=name,
                icon=folium.Icon(color="green", icon="briefcase")
            ).add_to(cluster)

            points.append([float(lat), float(lon)])

    if points:
        m.fit_bounds(points, padding=(20, 20))

    st_folium(m, height=420, width=None)
    if raw is None or len(raw) == 0:
        st.caption("Välj förening och storlek och klicka **Uppdatera förslag** för att visa markörer.")

# --------- Fördjupning & verktyg ---------
st.markdown("---")
st.subheader("Fördjupning & verktyg")

tab_stats, tab_export, tab_mail, tab_about = st.tabs(["📊 Statistik", "⬇️ Export", "✉️ Kontaktmall", "ℹ️ Om modellen"])

with tab_stats:
    raw = st.session_state.get("gg_results_raw")
    assoc_info = st.session_state.get("gg_assoc_info")

    if raw is None or len(raw) == 0:
        st.info("Inga resultat ännu – välj förening och storlek och klicka **Uppdatera förslag**.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Antal förslag", len(raw))
        if "distance_km" in raw:
            c2.metric("Snittavstånd", f"{raw['distance_km'].mean():.1f} km")
            c3.metric("Närmaste", f"{raw['distance_km'].min():.1f} km")
        else:
            c2.metric("Snittavstånd", "–")
            c3.metric("Närmaste", "–")
        c4.metric("Högsta score", f"{raw['score'].max():.2f}" if "score" in raw else "–")

        if assoc_info and "size_bin4" in assoc_info and "size_bin4" in raw.columns:
            diff = (raw["size_bin4"] - int(assoc_info["size_bin4"])).abs()
            match_rate = (diff <= 1).mean() * 100
            st.caption(f"Storleksmatch (±1 steg): **{match_rate:.0f}%**")

    if isinstance(raw, pd.DataFrame) and "distance_km" in raw.columns and raw["distance_km"].notna().any():
        st.markdown("**Fördelning av avstånd (km)**")
        edges = [0, 2, 5, 10, 20, 50, 100, 200, float("inf")]
        cats = pd.cut(raw["distance_km"].clip(lower=0.0), bins=edges, include_lowest=True, right=True)
        dist_df = (cats.value_counts().sort_index().rename_axis("Avståndsintervall").reset_index(name="Antal"))
        if not dist_df.empty:
            dist_df["Avståndsintervall"] = dist_df["Avståndsintervall"].astype(str)
            st.bar_chart(dist_df.set_index("Avståndsintervall")["Antal"])
        else:
            st.caption("Inga avståndsvärden att visa.")
    else:
        st.caption("Inga avståndsvärden att visa.")

with tab_export:
    raw = st.session_state.get("gg_results_raw")
    if raw is None or len(raw) == 0:
        st.info("Inga resultat att exportera ännu.")
    else:
        csv_bytes = raw.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Ladda ner resultat som CSV",
            data=csv_bytes,
            file_name="golden_goal_resultat.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("CSV-filen innehåller alla kolumner som visas i tabellen (och några till).")

with tab_mail:
    raw = st.session_state.get("gg_results_raw")
    assoc_info = st.session_state.get("gg_assoc_info")
    if raw is None or len(raw) == 0 or assoc_info is None:
        st.info("Generera först en lista med förslag för att skapa en kontaktmall.")
    else:
        company = st.selectbox("Välj företag att kontakta", raw["name"].astype(str).tolist())
        template = f"""Hej,

Jag representerar {assoc_info['name']} och undrar om vi kan diskutera ett möjligt samarbete/sponsoravtal.
Vi ser en god passform mellan er verksamhet och vår målgrupp.

Kort om oss:
• Förening: {assoc_info['name']}
• Plats (lat/lon): {assoc_info['lat']:.5f}, {assoc_info['lon']:.5f}
• Storlek (indikativ): {assoc_info.get('size_bin4', 'N/A')} (0–3)

Skulle ni ha möjlighet att ta ett kort samtal de närmaste dagarna?
Tack på förhand!

Vänliga hälsningar,
[Er signatur]
"""
        st.text_area("Brevutkast", template, height=220)
        st.caption("Kopiera och klistra in i e-post. (Automatisk sändning är inte aktiverad.)")

with tab_about:
    st.markdown("""
**Så funkar förslagen**

- Vi filtrerar först på **geografisk närhet** (adaptiv radie) och sedan på **företagsstorlek** (strikt mappning).
- Rankning väger **storlek 70 %** och **närhet 30 %**, så valet av 0–249 / 1000+ påverkar listan tydligt.
- Koordinater kvalitetssäkras (swap/SWEREF→WGS84) och saknade fixas för kartan via Nominatim.
""")

# --- Footer ---------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="color:#6b7280;font-size:0.95rem;">
      © 2025 Golden Goal · Byggd med Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
# --- End ------------------------------------------------------------------
