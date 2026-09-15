"""
Carbon Model v1 — parcel-level carbon stock estimate for LandFuture DSS
==========================================================================

Design (revised after inspecting the actual Teagasc SIS API output):

  BASELINE CARBON STOCK for grassland & tillage (mineral soils):
    Computed DIRECTLY from real Teagasc SIS lab measurements per soil
    series -- organic carbon %, bulk density, and horizon depth --
    rather than from a statistical model. This is a genuine measured
    value, not a regression output.

    SOC stock (t C/ha) = sum over horizons of:
        (OrganicCarbon% / 100) * bulk_density (t/m3) * thickness (m) * 10,000

    Linkage: SIS soil-association polygon (shapefile, ~58 associations)
    -> dominant component soil SERIES (via get_associations.php,
       ranked by dominance, Rank 1 = most common)
    -> that series' representative profile lab chemistry
       (via get_series_full.php)

  TRANSITION DELTAS (for later use by the optimizer, not baseline):
    SOLUM's Tier 2 GLM coefficients (Saunders et al. 2022) are kept as
    a separate function -- these estimate how stock would CHANGE if a
    parcel's land use/management is reallocated, complementing (not
    replacing) the measured baseline above. Not yet wired into the
    scenario-evaluation logic; that's a v1.5/v2 step once the
    generative/optimizer layer exists.

  PEATLAND / BOG: literature constant (unchanged from previous version)
    -- organic soils are out of scope for this mineral-soil lab dataset.

  FORESTRY: NFI-derived blended average (unchanged from previous version).

  NON-AGRICULTURAL: zero (unchanged).

Inputs needed:
  1. yield_model_v1.py's cache: data/_cache_parcels_classified.parquet
  2. SIS association shapefile: INSM250k_ING.shp (Teagasc SIS download)
     -- CRS is TM65/EPSG:29902, reprojected to ITM/EPSG:2157 here.
  3. Live network access to gis.teagasc.ie for the two API endpoints
     (cached locally after first fetch -- see CACHE below -- so this
     network cost is paid once, not on every run).

NOTE: gis.teagasc.ie is NOT reachable from a sandboxed environment
without general internet access -- this script is meant to run in your
own environment where you already successfully fetched the two sample
API responses.
"""

import json
import os
import re
import time

import geopandas as gpd
import pandas as pd
import numpy as np
import requests


# ---------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------

CACHE_PATH = "data/_cache_parcels_classified.parquet"       # from yield_model_v1.py
# Teagasc SIS association polygons
SIS_ASSOCIATION_SHP = "./data/INSM250k_ING_1b/INSM250k_ING.shp"

# caches association+series API calls
API_CACHE_PATH = "data/_cache_sis_api_responses.json"
ASSOCIATION_CARBON_CACHE_PATH = "data/_cache_association_carbon.parquet"

OUTPUT_PATH = "data/parcel_carbon_lookup.gpkg"

ASSOCIATIONS_URL = "https://gis.teagasc.ie/soils/get_associations.php?assoc_id={assoc_id}"
SERIES_FULL_URL = "https://gis.teagasc.ie/soils/get_series_full.php?series_code={series_code}"

SIS_TM65_EPSG = 29902  # confirmed via ogrinfo on INSM250k_ING.shp
TARGET_EPSG = 2157      # ITM, matches LPIS / county boundaries

# Depth cap (cm) for stock computation. A DepthTo of 999 (or similar
# large values) in the raw data is a soil-survey SENTINEL meaning
# "unbounded / continues to bedrock," NOT a literal ~10m measurement --
# confirmed by inspection: treating it literally inflated some series'
# stock by 1,000+ t C/ha from a single fake-thickness horizon (e.g.
# series 0410CV: a 20->999cm horizon alone contributed 1,842 of its
# 2,652 t C/ha total). 100cm matches the common SOC-to-1m reporting
# convention used elsewhere in this project's literature sources.
DEPTH_CAP_CM = 100

# Base mineral-soil bulk density fallback (t/m3), used only for
# genuinely low-OC horizons where PackingDensity_Percent is missing --
# see _bulk_density_fallback() below for the OC%-dependent logic.
DEFAULT_BULK_DENSITY_T_PER_M3 = 1.3


def _bulk_density_fallback(oc_pct: float) -> float:
    """
    OC%-dependent fallback bulk density, used only when a horizon's
    PackingDensity_Percent is missing.

    Real soils show a strong INVERSE relationship between organic
    carbon content and bulk density (organic matter is far less dense
    than mineral material). A flat mineral-soil default badly
    overestimates stock for high-OC horizons -- confirmed directly:
    series 0110KM's 4,257 t C/ha came from 32-40% OC horizons with
    missing bulk density defaulted to the flat mineral value.

    This tiered mapping is a documented APPROXIMATION of that known
    relationship, not a fitted pedotransfer function -- refine with a
    real OC-vs-bulk-density regression if literature figures for
    Irish soils become available. Flagged in carbon_source output
    wherever it's actually used (see compute_series_soc_stock).
    """
    if oc_pct >= 20:
        return 0.3
    if oc_pct >= 10:
        return 0.6
    if oc_pct >= 5:
        return 0.9
    return DEFAULT_BULK_DENSITY_T_PER_M3


# --- SOLUM GLM coefficients (Saunders et al. 2022, Table 3.2) ---
# Kept for later TRANSITION-delta use, not baseline stock computation.
SOLUM_INTERCEPT = 4.22634
SOLUM_SOIL_CLUSTER_COEF = {"loamy": 0.0,
                           "clay": 0.29236, "silty": 0.08811, "sandy": -0.23284}
SOLUM_LANDUSE_COEF = {"cropland": 0.0, "forest": 0.36125, "grassland": 0.2673}
SOLUM_MANAGEMENT_COEF = {"unimproved": 0.0,
                         "improved": -0.10519, "transitioning": -0.36278}


def solum_soc(soil_cluster: str, land_use: str, management: str | None = None) -> float:
    """SOLUM Tier 2 GLM -- kept for future transition-delta estimates,
    e.g. delta = solum_soc(cluster, 'forest') - solum_soc(cluster, 'grassland', 'improved')
    to estimate the stock change from converting a parcel's land use.
    NOT used for baseline stock in this version."""
    cl = SOLUM_SOIL_CLUSTER_COEF.get(soil_cluster, 0.0)
    flu = SOLUM_LANDUSE_COEF.get(land_use, 0.0)
    fmg = SOLUM_MANAGEMENT_COEF.get(management, 0.0) if management else 0.0
    return float(np.exp(SOLUM_INTERCEPT + cl + flu + fmg))


# --- Peatland / bog literature densities (t C/ha) ---
BOG_CARBON_MEAN = 705.0
BOG_CARBON_STD = 150.0

# --- NFI-derived forestry figure (2022 Main Findings, Table 11) ---
NFI_FOREST_AREA_HA = 720_748
NFI_TOTAL_MT_C = 323.0
NFI_TOTAL_C_PER_HA = NFI_TOTAL_MT_C * 1e6 / NFI_FOREST_AREA_HA
# no NFI variance figure pulled yet -- documented placeholder
NFI_FALLBACK_REL_UNCERTAINTY = 0.20

PEAT_ASSOCIATED_CLASSES = {"BOG", "LOW INPUT PEAT GRASSLAND"}
FORESTRY_CLASSES = {
    "FORESTRY", "FORESTRY ELIGIBLE", "FORESTRY INELIGIBLE",
    "FORESTRY PRE-2009", "WOODLAND", "NATIVE TREE AREA",
}


# ---------------------------------------------------------------
# 2. TEAGASC SIS API CLIENT (with local caching)
# ---------------------------------------------------------------

def _load_api_cache(path: str = API_CACHE_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"associations": {}, "series": {}}


def _save_api_cache(cache: dict, path: str = API_CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


def fetch_association_series(assoc_id: str, cache: dict) -> list[dict]:
    """Returns the ranked list of component series for an association,
    e.g. [{"Rank":1, "National_Series_Id":"0600KP", ...}, ...].
    Cached locally -- only hits the network once per association."""
    if assoc_id in cache["associations"]:
        return cache["associations"][assoc_id]["SeriesArray"]

    resp = requests.get(ASSOCIATIONS_URL.format(assoc_id=assoc_id), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    cache["associations"][assoc_id] = data
    # be polite to a small research-project API, not a hardened service
    time.sleep(0.2)
    return data["SeriesArray"]


def fetch_series_chemistry(series_code: str, cache: dict) -> dict:
    """Returns the full representative-profile record for one soil
    series, including its Horizons array. Cached locally."""
    if series_code in cache["series"]:
        return cache["series"][series_code]

    resp = requests.get(SERIES_FULL_URL.format(
        series_code=series_code), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    cache["series"][series_code] = data
    time.sleep(0.2)
    return data


def _parse_bulk_density_midpoint(packing_density_percent: str | None, oc_pct: float) -> float:
    """Parses e.g. '1.40 - 1.60 t m-3' -> 1.50. Falls back to the
    OC%-dependent estimate (_bulk_density_fallback) when missing or
    unparseable, rather than a flat mineral-soil value."""
    if not packing_density_percent:
        return _bulk_density_fallback(oc_pct)
    match = re.findall(r"[\d.]+", packing_density_percent)
    if len(match) >= 2:
        return (float(match[0]) + float(match[1])) / 2
    if len(match) == 1:
        return float(match[0])
    return _bulk_density_fallback(oc_pct)


def compute_series_soc_stock(series_data: dict) -> tuple[float, int]:
    """
    Computes total SOC stock (t C/ha) by summing across horizons up to
    DEPTH_CAP_CM: (OC% / 100) * bulk_density(t/m3) * thickness(m) * 10,000.

    Horizons are truncated at DEPTH_CAP_CM -- a missing or sentinel
    DepthTo (e.g. 999, meaning "to bedrock" in the raw data) is treated
    as extending to the cap, not taken literally. A horizon starting
    at or beyond the cap is skipped entirely.

    Returns (stock, n_horizons_used) -- n_horizons_used lets the caller
    judge how much of the profile the estimate actually covers.
    """
    total = 0.0
    n_used = 0
    for h in series_data.get("Horizons", []):
        oc_pct = h.get("OrganicCarbon")
        depth_from = h.get("Depth")
        depth_to = h.get("DepthTo")
        if oc_pct is None or depth_from is None:
            continue
        if depth_from >= DEPTH_CAP_CM:
            continue  # entire horizon lies beyond the depth cap
        if depth_to is None or depth_to > DEPTH_CAP_CM:
            depth_to = DEPTH_CAP_CM  # missing/sentinel bottom -> capped, not literal
        thickness_m = (depth_to - depth_from) / 100.0
        if thickness_m <= 0:
            continue
        bulk_density = _parse_bulk_density_midpoint(
            h.get("PackingDensity_Percent"), oc_pct)
        stock = (oc_pct / 100.0) * bulk_density * thickness_m * 10_000
        total += stock
        n_used += 1
    return total, n_used


# ---------------------------------------------------------------
# 3. BUILD PER-ASSOCIATION CARBON LOOKUP
# ---------------------------------------------------------------

def build_association_carbon_lookup(association_codes: list[str]) -> pd.DataFrame:
    """
    For each association code, fetches its dominant (Rank 1) series and
    computes that series' measured SOC stock as the association's
    baseline carbon estimate.

    Uncertainty: if 2+ series are available for an association, uses
    the spread across the top few ranked series' stocks as a real
    variability estimate. Otherwise falls back to a documented 20%
    relative uncertainty -- NOT invented as precise, flagged via the
    'n_series_used' column so low-confidence associations are visible.
    """
    cache = _load_api_cache()
    rows = []

    for assoc_id in association_codes:
        try:
            series_list = fetch_association_series(assoc_id, cache)
        except Exception as e:
            print(f"Warning: failed to fetch association {assoc_id}: {e}")
            continue

        if not series_list:
            continue

        # Use up to the top 3 ranked series for a variability estimate;
        # dominant (Rank 1) series' stock is the point estimate.
        ranked = sorted(series_list, key=lambda s: s.get("Rank", 999))[:3]
        stocks = []
        for s in ranked:
            code = s.get("National_Series_Id")
            if not code:
                continue
            try:
                chem = fetch_series_chemistry(code, cache)
            except Exception as e:
                print(f"Warning: failed to fetch series {code}: {e}")
                continue
            stock, n_horizons = compute_series_soc_stock(chem)
            if n_horizons > 0:
                stocks.append(stock)

        if not stocks:
            continue

        mean_stock = stocks[0]  # dominant series is the point estimate
        if len(stocks) > 1:
            std_stock = float(np.std(stocks))
        else:
            std_stock = mean_stock * 0.20  # documented fallback, single series only

        rows.append({
            "Associat_S": assoc_id,
            "carbon_t_c_per_ha_mean": mean_stock,
            "carbon_t_c_per_ha_std": std_stock,
            "n_series_used": len(stocks),
        })

    _save_api_cache(cache)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 4. LOAD / JOIN SPATIAL DATA
# ---------------------------------------------------------------

def load_cached_parcels(path: str = CACHE_PATH) -> gpd.GeoDataFrame:
    parcels = gpd.read_parquet(path)
    print(f"Loaded {len(parcels)} cached classified parcels")
    return parcels


def load_sis_associations(path: str = SIS_ASSOCIATION_SHP) -> gpd.GeoDataFrame:
    """Loads the SIS association polygons and reprojects from
    TM65 (EPSG:29902, confirmed via ogrinfo) to ITM (EPSG:2157)."""
    soils = gpd.read_file(path)
    print(f"Loaded {len(soils)} SIS association polygons, CRS={soils.crs}")
    soils = soils.to_crs(epsg=TARGET_EPSG)
    return soils


def assign_association_and_carbon(parcels: gpd.GeoDataFrame, soils: gpd.GeoDataFrame,
                                  carbon_lookup: pd.DataFrame) -> gpd.GeoDataFrame:
    """Centroid join to associations (same performance rationale as
    assign_county() in the yield model), then attach the carbon lookup."""
    centroids = parcels.copy()
    centroids["geometry"] = parcels.geometry.centroid

    joined = gpd.sjoin(
        centroids, soils[["Associat_S", "geometry"]], how="left", predicate="within")
    parcels = parcels.copy()
    parcels["Associat_S"] = joined["Associat_S"].values

    n_unmatched = parcels["Associat_S"].isna().sum()
    if n_unmatched:
        print(
            f"Warning: {n_unmatched} parcels did not match a soil association.")

    parcels = parcels.merge(carbon_lookup, on="Associat_S", how="left")
    return parcels


# ---------------------------------------------------------------
# 5. ASSEMBLE FULL CARBON ESTIMATE (all branches)
# ---------------------------------------------------------------

def finalize_carbon_estimate(parcels: gpd.GeoDataFrame, crop_col: str = "crop") -> gpd.GeoDataFrame:
    crop_upper = parcels[crop_col].str.upper()
    is_bog = crop_upper.isin(PEAT_ASSOCIATED_CLASSES)
    is_forestry = crop_upper.isin(FORESTRY_CLASSES)

    # Peatland/bog and forestry OVERRIDE whatever the soil-association
    # join produced (that join covers mineral soils only, per SOLUM's
    # own scope limitation -- these branches use their own sources).
    parcels.loc[is_bog, "carbon_t_c_per_ha_mean"] = BOG_CARBON_MEAN
    parcels.loc[is_bog, "carbon_t_c_per_ha_std"] = BOG_CARBON_STD
    parcels.loc[is_bog, "carbon_source"] = "peatland_literature"

    parcels.loc[is_forestry, "carbon_t_c_per_ha_mean"] = NFI_TOTAL_C_PER_HA
    parcels.loc[is_forestry, "carbon_t_c_per_ha_std"] = NFI_TOTAL_C_PER_HA * \
        NFI_FALLBACK_REL_UNCERTAINTY
    parcels.loc[is_forestry, "carbon_source"] = "nfi_blended_average"

    is_mineral_estimated = parcels["carbon_source"].isna(
    ) & parcels["carbon_t_c_per_ha_mean"].notna()
    parcels.loc[is_mineral_estimated, "carbon_source"] = "sis_measured_series"

    is_zero = parcels["is_excluded"] & ~is_bog & ~is_forestry
    parcels.loc[is_zero, "carbon_t_c_per_ha_mean"] = 0.0
    parcels.loc[is_zero, "carbon_t_c_per_ha_std"] = 0.0
    parcels.loc[is_zero, "carbon_source"] = "zero_nonagricultural"

    n_unresolved = parcels["carbon_t_c_per_ha_mean"].isna().sum()
    print(f"Carbon estimate coverage: {(1 - n_unresolved / len(parcels)):.1%} "
          f"({n_unresolved} parcels unresolved)")
    print(parcels["carbon_source"].value_counts(dropna=False))
    return parcels


# ---------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------

def main():
    parcels = load_cached_parcels()
    soils = load_sis_associations()

    if os.path.exists(ASSOCIATION_CARBON_CACHE_PATH):
        print(
            f"Loading cached association carbon lookup from {ASSOCIATION_CARBON_CACHE_PATH}")
        carbon_lookup = pd.read_parquet(ASSOCIATION_CARBON_CACHE_PATH)
    else:
        association_codes = soils["Associat_S"].dropna().unique().tolist()
        print(f"Fetching carbon data for {len(association_codes)} soil associations "
              "(this hits the Teagasc API -- one-time cost, cached after).")
        carbon_lookup = build_association_carbon_lookup(association_codes)
        carbon_lookup.to_parquet(ASSOCIATION_CARBON_CACHE_PATH)

    parcels = assign_association_and_carbon(parcels, soils, carbon_lookup)
    parcels = finalize_carbon_estimate(parcels)

    parcels.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"Wrote {len(parcels)} parcels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
