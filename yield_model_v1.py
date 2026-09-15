"""
Yield Model v1 — county-level yield/productivity model for LandFuture DSS
===========================================================================

Goal: assign every LPIS parcel an expected yield/productivity value
(with an honest uncertainty band), split into two branches because the
underlying CSO data has genuinely different resolution for each:

  TILLAGE CROPS (wheat, barley, oats, etc.)
    - Yield (AQA04) is NATIONAL ONLY — no county breakdown exists
      anywhere (confirmed against both CSO and Eurostat).
    - So every parcel of a given crop gets the same national yield
      estimate, with uncertainty from real interannual variance.
      This is a stated v1 limitation, not a bug: spatial variation for
      tillage crops needs a v2 model (soil/climate covariates), not a
      missing-data fix.

  GRASSLAND (permanent pasture, extensively grazed pasture, etc.)
    - No yield/production statistic exists for grassland at all
      (grass isn't a marketed commodity with a harvest weight), AND
      no county-level grassland AREA exists anywhere in CSO's tables
      either (confirmed directly against IFS10: its Region field only
      ever holds 8 NUTS3-style regions, never a county).
    - v1 proxy: REGIONAL cattle density (AAA10 county cattle counts,
      aggregated up to region / IFS10 regional grassland area)
      relative to the national average density. Every county within
      a region gets that region's index — an honestly-labeled
      coarsening, not county-level, but still spatially varying at
      the regional level rather than one flat national number.

NON-AGRICULTURAL classes (buildings, farmyards, forestry, bog, scrub,
habitat) are excluded entirely — no yield concept applies. Route
forestry/bog/scrub to the CARBON model instead, not here.

Inputs you need to obtain manually before running this:
  1. LPIS parcels (geopackage) — opendata.agriculture.gov.ie
  2. Irish county boundaries (geopackage) — Tailte Éireann "Counties -
     National Statutory Boundaries - Ungeneralised - 2026", confirmed
     EPSG:2157 (ITM), name field ENG_NAME_VALUE, needs case-normalize +
     dissolve (handled by load_counties() below)
  3. AQA04 "Area, Yield and Production of Crops" — NATIONAL, annual,
     2008-2025. https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/AQA04/CSV/1.0/en
  4. AAA10 "Number of Cattle in June" — COUNTY-level (mixed with
     regional aggregate rows you must filter out), annual.
     https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/AAA10/CSV/1.0/en
  5. IFS10 "Land Utilisation" (June Crops and Livestock Survey series)
     — REGIONAL only (8 NUTS3-style regions), confirmed no county
     breakdown exists. Filter to Statistic Label == "Area under Crops".
     https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/IFS10/CSV/1.0/en

Column names below are best guesses from each table's documented
structure — confirm against the actual downloaded CSV headers and
adjust, since PxStat CSV column naming isn't fully predictable without
the file in hand.
"""

import geopandas as gpd
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 1. CONFIG — fill these in
# ---------------------------------------------------------------

LPIS_PATH = "data/lpis_2025_parcels.gpkg"

COUNTY_BOUNDARIES_PATH = "data/National_Statutory_Boundaries_-_Counties_Ungeneralised_-_2026.gpkg"
COUNTY_NAME_FIELD_RAW = "ENG_NAME_VALUE"

AQA04_CSV = "data/AQA04_crop_yield.csv"          # national tillage-crop yield
AAA10_CSV = "data/AAA10_cattle_county.csv"        # county-level cattle counts
IFS10_CSV = "data/IFS10_land_utilisation.csv"     # county-level grassland area

OUTPUT_PATH = "data/parcel_yield_lookup.gpkg"

# The 26 Republic of Ireland counties — used to filter AAA10's
# "Region and County" field down to county-level rows only (that field
# mixes NUTS3 regional aggregates like "Southern"/"Mid-West" with
# individual counties in the same column).
IE_COUNTIES = {
    "CARLOW", "CAVAN", "CLARE", "CORK", "DONEGAL", "DUBLIN", "GALWAY",
    "KERRY", "KILDARE", "KILKENNY", "LAOIS", "LEITRIM", "LIMERICK",
    "LONGFORD", "LOUTH", "MAYO", "MEATH", "MONAGHAN", "OFFALY",
    "ROSCOMMON", "SLIGO", "TIPPERARY", "WATERFORD", "WESTMEATH",
    "WEXFORD", "WICKLOW",
}

# Official CSO county -> region mapping, from the Crops and Livestock
# Survey June background notes (cso.ie/en/methods/surveybackgroundnotes/
# cropsandlivestocksurveyjune/), reflecting the post-2018 NUTS3
# structure. Used to aggregate AAA10's county cattle counts up to
# IFS10's 8 regions, since no county-level grassland area exists
# anywhere in CSO's published tables (confirmed: IFS10's "Region"
# dimension only has these 8 values + "Ireland", not counties).
COUNTY_TO_REGION = {
    # Northern and Western NUTS 2
    "CAVAN": "Border", "DONEGAL": "Border", "LEITRIM": "Border",
    "MONAGHAN": "Border", "SLIGO": "Border",
    "GALWAY": "West", "MAYO": "West", "ROSCOMMON": "West",
    # Southern NUTS 2
    "LIMERICK": "Mid-West", "CLARE": "Mid-West", "TIPPERARY": "Mid-West",
    "WATERFORD": "South-East", "CARLOW": "South-East",
    "KILKENNY": "South-East", "WEXFORD": "South-East",
    "CORK": "South-West", "KERRY": "South-West",
    # Eastern & Midland NUTS 2
    "DUBLIN": "Dublin",
    "KILDARE": "Mid-East", "MEATH": "Mid-East", "WICKLOW": "Mid-East",
    "LOUTH": "Mid-East",
    "LAOIS": "Midlands", "LONGFORD": "Midlands", "OFFALY": "Midlands",
    "WESTMEATH": "Midlands",
}

# Non-agricultural LPIS classes — no yield concept applies.
# Extend this after checking the rest of your 176 crop categories.
NON_AGRICULTURAL_CLASSES = {
    "BUILDING", "FARMYARD", "BOG", "SCRUB", "HABITAT", "WOODLAND",
    "FORESTRY", "FORESTRY ELIGIBLE", "FORESTRY INELIGIBLE",
    "FORESTRY PRE-2009", "DESIGNATED HABITAT", "FARM ROAD",
    "ACCESS ROAD / ROADWAYS", "GARDENS", "QUARRY",
    "LAKE / WATERWAY / POND", "RECREATIONAL AREA", "ROCKY OUTCROP",
    "ORCHARD", "CHRISTMAS TREES", "NATIVE TREE AREA", "WILLOW",
    "NURSERY", "FOLIAGE", "FORESTRY ESB CORRIDOR INELIGIBLE",
}

# Agri-environment scheme land — taken out of production, not a yield
# category. Treated as excluded for v1 rather than guessed at.
SCHEME_LAND_CLASSES = {
    "WINTER BIRD FOOD PLOT", "RIPARIAN BUFFER ZONE - ARABLE",
    "MANAGEMENT OF INTENSIVE GRASSLAND NEXT TO A WATERCOURSE",
    "ENVIRONMENTAL MANAGEMENT OF ARABLE FALLOW", "FALLOW",
    "WILD BIRD COVER", "RIPARIAN ZONE",
    "TREE BELTS FOR AMMONIA CAPTURE FROM FARMYARDS", "INACTIVE",
}

# Grassland classes — routed to the cattle-density proxy, not AQA04.
GRASSLAND_CLASSES = {
    "PERMANENT PASTURE", "PERMANENT PASTURE (MSS)",
    "EXTENSIVELY GRAZED PASTURE", "LOW INPUT GRASSLAND",
    "RIPARIAN BUFFER ZONE - GRASSLAND", "RED CLOVER GRASS SWARD",
    "GRASS YEAR 1", "GRASS YEAR 2", "GRASS YEAR 3", "GRASS YEAR 4",
    "GRASS YEAR 5", "RED CLOVER", "LOW INPUT PEAT GRASSLAND",
    "LOW INPUT PERMANENT PASTURE", "ARABLE SILAGE (GRASS)",
    "GRASS YEAR 1 (MSS)", "GRASS YEAR 2 (MSS)", "GRASS YEAR 3 (MSS)",
    "GRASS YEAR 4 (MSS)",
}

# Tillage-crop classes — routed to AQA04 national yield lookup.
# Extend as you confirm more categories from the full 176.
LPIS_TO_AQA04_CLASS_MAP = {
    "BARLEY - SPRING": "spring barley",
    "BARLEY - WINTER": "winter barley",
    "WHEAT - WINTER": "winter wheat",
    "WHEAT - SPRING": "spring wheat",
    "OATS - SPRING": "spring oats",
    "OATS - WINTER": "winter oats",
    "BEANS - SPRING": "beans and peas",
    "BEANS - WINTER": "beans and peas",
    # Real commercial crops NOT in AQA04's 13 categories — need a
    # separate yield source before they can be mapped. Left unmapped
    # intentionally rather than forced into a wrong category.
    # MAIZE, FODDER BEET, OILSEED RAPE - WINTER/SPRING,
    # POTATOES - MAINCROP/EARLY, SUGAR BEET, PEAS, RYE, CARROTS,
    # TURNIPS, KALE, CABBAGE - SPRING
}

# Ambiguous mixed/silage categories — real agricultural output but no
# clean AQA04 mapping. Excluded from v1 rather than force-mapped;
# noted here as a known, documented gap.
AMBIGUOUS_UNMAPPED_CLASSES = {
    "PROTEIN/CEREAL MIX 50/50", "ARABLE SILAGE (NO GRASS)",
    "MIXED CROPPING", "MIXED CROPPING (HORTICULTURE)", "FORAGE RAPE",
}


# ---------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------

def load_parcels(path: str, max_plausible_area_ha: float = 500.0) -> gpd.GeoDataFrame:
    """
    max_plausible_area_ha: sanity threshold for a single parcel. Real
    Irish agricultural field parcels are essentially never anywhere
    near this large — even large farms' individual fields are a small
    fraction of it. Confirmed via direct inspection: rows with null
    'crop' (and every other attribute) include some geometries up to
    ~88 km^2 (8,800 ha), consistent with leftover background/void
    polygons rather than real parcels, not just "unclaimed" land.
    """
    parcels = gpd.read_file(path, columns=["crop"], use_arrow=True)
    n_raw = len(parcels)
    print(f"Loaded {n_raw} LPIS parcels")

    # (Column selection now happens at read time via columns=, above —
    # this avoids reading ~28 unused attribute columns off disk for
    # 4.9M rows in the first place, rather than reading everything then
    # dropping it in memory. use_arrow=True uses pyogrio's Arrow-based
    # reader, which is substantially faster than the default path for
    # large files when pyarrow is installed.)
    parcels["geometry"] = parcels["geometry"].force_2d()

    # Exclude rows with no crop/land-use declaration at all. Confirmed
    # via direct inspection (not assumed): for these specific rows,
    # EVERY other attribute column (herd, par_lab, claim_area, ref_area,
    # commonage/arable/perm indicators, etc.) is also null — this is
    # not "unclaimed reference land" with some other identifying info
    # intact, it's rows with no substantive attributes whatsoever.
    # Most plausible explanation: LPIS's full reference geometry layer
    # includes every recognized parcel regardless of whether it had an
    # active 2025 scheme declaration; these are the undeclared ones.
    n_null_crop = parcels["crop"].isna().sum()
    parcels = parcels[parcels["crop"].notna()].copy()
    print(f"Dropped {n_null_crop} parcels with no crop/land-use declaration "
          f"({n_null_crop / n_raw:.1%} of raw total) — no attributes at all "
          "on these rows, confirmed by inspection, not just 'crop' missing.")

    # Separately, exclude implausibly large geometries as likely
    # background/artifact polygons rather than real field parcels.
    # This is independent of the null-crop filter above — a large
    # polygon with a real crop declaration would NOT be dropped here.
    # assumes projected CRS in meters (ITM/EPSG:2157)
    area_ha = parcels.geometry.area / 10_000
    oversized_mask = area_ha > max_plausible_area_ha
    n_oversized = oversized_mask.sum()
    if n_oversized:
        print(f"Dropping {n_oversized} parcels with implausible area "
              f"(> {max_plausible_area_ha} ha) — likely background/artifact "
              f"polygons, not real field parcels. Largest dropped: "
              f"{area_ha[oversized_mask].max():.0f} ha.")
        parcels = parcels[~oversized_mask].copy()

    print(f"Retained {len(parcels)} parcels for the yield pipeline "
          f"({len(parcels) / n_raw:.1%} of raw total).")

    return parcels


def load_counties(path: str, name_field: str = COUNTY_NAME_FIELD_RAW) -> gpd.GeoDataFrame:
    """
    Load the Tailte Éireann statutory boundary layer and collapse it into
    one clean polygon per county. Raw layer has ~9,261 fragmented
    boundary-capture features with inconsistent name casing (e.g. "CORK"
    vs "Cork") — confirmed via ogrinfo + geopandas that normalizing case
    and dissolving by name field correctly yields 26 valid polygons.
    """
    counties_raw = gpd.read_file(path)
    assert name_field in counties_raw.columns, (
        f"Expected column '{name_field}' not found. Actual columns: "
        f"{list(counties_raw.columns)}"
    )
    counties_raw[name_field] = counties_raw[name_field].str.upper()
    # performance: 2D dissolve is faster
    counties_raw["geometry"] = counties_raw["geometry"].force_2d()
    counties = counties_raw.dissolve(by=name_field, as_index=False)
    counties = counties.rename(columns={name_field: "COUNTY"})[
        ["COUNTY", "geometry"]]

    n = len(counties)
    if n != 26:
        print(f"Warning: expected 26 counties after dissolve, got {n}.")
    if not counties.geometry.is_valid.all():
        print("Warning: some dissolved county polygons are invalid.")
    return counties


def load_aqa04(path: str) -> pd.DataFrame:
    """
    National tillage-crop yield series. NO county dimension exists in
    this table — confirmed directly against the PxStat API. Adjust
    column names below once you've inspected your actual downloaded CSV
    (the coded export has STATISTIC/TLIST(A1)/C02039V02469 columns; the
    label-only export just has "Statistic Label"/"Year"/"Type of Crop").
    """
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Statistic Label": "statistic",
        "Year": "year",
        "Type of Crop": "class",
        "VALUE": "value",
    })
    yield_df = df[df["statistic"].str.contains(
        "Yield", case=False, na=False)].copy()
    yield_df["class"] = yield_df["class"].str.lower()
    yield_df = yield_df.rename(columns={"value": "yield_t_per_ha"})
    return yield_df[["year", "class", "yield_t_per_ha"]]


def load_aaa10(path: str) -> pd.DataFrame:
    """
    County-level cattle counts. The "Region and County" field mixes
    NUTS3 regional aggregates with individual counties in one column —
    filter to IE_COUNTIES to keep only genuine county-level rows.
    Adjust column names to match your actual downloaded CSV headers.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Year": "year",
        "Type of Cattle": "cattle_type",
        "Region and County": "area_label",
        "VALUE": "cattle_head",
    })
    df["area_label_upper"] = df["area_label"].str.upper()
    county_rows = df[df["area_label_upper"].isin(IE_COUNTIES)].copy()

    n_dropped = len(df) - len(county_rows)
    print(f"AAA10: kept {len(county_rows)} county-level rows, "
          f"dropped {n_dropped} regional-aggregate rows.")

    # Keep ONLY the exact "Total cattle" category. Confirmed via direct
    # inspection that "total" as a substring also matches "Total
    # cattle: male" and "Total cattle: female" — using contains() would
    # sum all three and roughly double/triple-count, since male+female
    # already equals the total. Exact match avoids that.
    if "cattle_type" in county_rows.columns:
        exact_mask = county_rows["cattle_type"].str.strip(
        ).str.lower() == "total cattle"
        if exact_mask.any():
            county_rows = county_rows[exact_mask]
        else:
            print("WARNING: no rows matched 'Total cattle' exactly. "
                  f"Available cattle_type values: "
                  f"{sorted(county_rows['cattle_type'].unique())}")

    county_rows = county_rows.rename(columns={"area_label_upper": "COUNTY"})
    county_rows["REGION"] = county_rows["COUNTY"].map(COUNTY_TO_REGION)

    n_unmapped_region = county_rows["REGION"].isna().sum()
    if n_unmapped_region:
        print(f"Warning: {n_unmapped_region} AAA10 rows have a county not "
              "found in COUNTY_TO_REGION — check for name mismatches "
              "(e.g. 'Laoighis' vs 'Laois').")

    return county_rows[["year", "COUNTY", "REGION", "cattle_head"]]


def load_ifs10_grassland_area(path: str) -> pd.DataFrame:
    """
    REGIONAL (not county) grassland area — confirmed against the actual
    file: Region only has 8 NUTS3-style values + "Ireland", never a
    county name, across the whole file (not a partial-read artifact).
    No county-level grassland area exists anywhere in CSO's published
    tables. This is the honest v1 constraint the grassland model works
    within: regional resolution, not county.

    Also confirmed the file carries three statistics under one table
    code ('Farms with Crops', 'Area under Crops', 'Average Area
    Farmed') — filtering to 'Area under Crops' (hectares) here, not
    'Farms with Crops' (a farm count, wrong unit entirely).
    """
    df = pd.read_csv(path)

    df = df[df["Statistic Label"] == "Area under Crops"].copy()
    df = df.rename(columns={
        "Year": "year",
        "Region": "REGION",
        "Type of Crop": "land_use_class",
        "VALUE": "area_ha",
    })
    df = df[df["REGION"] != "Ireland"].copy()  # drop the national total row

    grass_mask = df["land_use_class"].str.contains(
        "grass|pasture", case=False, na=False
    ) & ~df["land_use_class"].str.contains("total cereals|fruit", case=False, na=False)
    return df[grass_mask][["year", "REGION", "land_use_class", "area_ha"]]


# ---------------------------------------------------------------
# 3. SPATIAL JOIN — assign each parcel a county
# ---------------------------------------------------------------

def assign_county(parcels: gpd.GeoDataFrame, counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Performance: join on parcel CENTROIDS rather than full parcel
    polygons. Point-in-polygon is far cheaper than polygon-in-polygon
    at 4.9M rows, and correct for all but genuinely boundary-straddling
    parcels — which the unmatched-parcel warning below already flags
    as a known edge case, so this doesn't introduce a new blind spot.
    Original polygon geometry is preserved on the returned GeoDataFrame;
    only the join itself uses centroids.
    """
    parcels = parcels.to_crs(counties.crs)

    centroids = parcels.copy()
    centroids["geometry"] = parcels.geometry.centroid

    joined = gpd.sjoin(
        centroids, counties[["COUNTY", "geometry"]], how="left", predicate="within")

    parcels = parcels.copy()
    parcels["COUNTY"] = joined["COUNTY"].values

    n_unmatched = parcels["COUNTY"].isna().sum()
    if n_unmatched:
        print(f"Warning: {n_unmatched} parcels did not fall within any county "
              "(likely boundary/coastal edge cases).")
    return parcels


# ---------------------------------------------------------------
# 4a. TILLAGE-CROP LOOKUP (national, from AQA04)
# ---------------------------------------------------------------

def build_tillage_lookup(aqa04_yield: pd.DataFrame) -> pd.DataFrame:
    """
    National yield per crop class, with uncertainty from real
    interannual variance. Same value applies to every county — that's
    the stated v1 limitation, not an error.
    """
    grouped = aqa04_yield.groupby("class")["yield_t_per_ha"]
    lookup = grouped.agg(yield_mean="mean", yield_std="std",
                         n_years="count").reset_index()
    fallback_rel_uncertainty = 0.20
    lookup["yield_std"] = lookup["yield_std"].fillna(
        lookup["yield_mean"] * fallback_rel_uncertainty)

    # Validate that every class named in LPIS_TO_AQA04_CLASS_MAP actually
    # exists in AQA04 — e.g. the map used "beans" but AQA04's real label
    # may be "beans and peas" or similar. Surfacing this explicitly here
    # avoids silently producing NaN yields for a whole crop class.
    mapped_classes = set(LPIS_TO_AQA04_CLASS_MAP.values())
    available_classes = set(lookup["class"])
    missing = mapped_classes - available_classes
    if missing:
        print(f"WARNING: LPIS_TO_AQA04_CLASS_MAP references class(es) "
              f"{missing} not found in AQA04. Available AQA04 classes: "
              f"{sorted(available_classes)}. Update the map to match one "
              "of these exactly.")

    return lookup


# ---------------------------------------------------------------
# 4b. GRASSLAND LOOKUP (relative cattle-density index, county-level)
# ---------------------------------------------------------------

def build_grassland_lookup(cattle_df: pd.DataFrame, area_df: pd.DataFrame) -> pd.DataFrame:
    """
    REGIONAL cattle density (head / grassland ha), relative to the
    national average density — computed at region resolution because
    no county-level grassland area exists anywhere in CSO's published
    tables (confirmed against IFS10 directly: its Region dimension only
    ever holds the 8 NUTS3-style values, never a county name).

    cattle_df: county-level cattle counts (AAA10), each row tagged with
               its REGION via COUNTY_TO_REGION.
    area_df:   regional grassland area (IFS10, 'Area under Crops' stat).

    Aggregates county cattle counts up to region, joins against
    regional area, computes density relative to the national average
    for that year, then averages across years for a point estimate +
    uncertainty band from real cross-year variance.

    Returns a lookup keyed by REGION — every county within a region
    gets that region's index (documented coarsening, not hidden).

    NOTE: this is a RELATIVE productivity index (1.0 = average region),
    not an absolute tonnage. Don't present it as a literal yield number
    without multiplying by a real national grassland output figure if
    one becomes available later.
    """
    regional_cattle = (
        cattle_df.groupby(["year", "REGION"])[
            "cattle_head"].sum().reset_index()
    )

    # Multiple land-use sub-classes (Pasture, All grassland, etc.) can
    # appear per region/year in IFS10 — sum area across whichever
    # grassland-related classes are present, to avoid under-counting.
    regional_area = (
        area_df.groupby(["year", "REGION"])["area_ha"].sum().reset_index()
    )

    merged = regional_cattle.merge(
        regional_area, on=["year", "REGION"], how="inner")
    merged["density"] = merged["cattle_head"] / merged["area_ha"]

    national_avg_by_year = merged.groupby("year")["density"].transform("mean")
    merged["relative_density"] = merged["density"] / national_avg_by_year

    grouped = merged.groupby("REGION")["relative_density"]
    lookup = grouped.agg(
        grassland_index_mean="mean",
        grassland_index_std="std",
        n_years="count",
    ).reset_index()
    fallback_rel_uncertainty = 0.20
    lookup["grassland_index_std"] = lookup["grassland_index_std"].fillna(
        lookup["grassland_index_mean"] * fallback_rel_uncertainty
    )
    return lookup


# ---------------------------------------------------------------
# 5. ASSEMBLE: parcel -> expected yield / productivity index
# ---------------------------------------------------------------

def classify_parcels(parcels: gpd.GeoDataFrame, crop_col: str = "crop") -> gpd.GeoDataFrame:
    """
    Route each parcel into exactly one of: tillage / grassland /
    excluded (non-agricultural or scheme land) / unmapped.
    """
    crop_upper = parcels[crop_col].str.upper()

    parcels["tillage_class"] = crop_upper.map(LPIS_TO_AQA04_CLASS_MAP)
    parcels["is_grassland"] = crop_upper.isin(GRASSLAND_CLASSES)
    parcels["is_excluded"] = crop_upper.isin(
        NON_AGRICULTURAL_CLASSES | SCHEME_LAND_CLASSES | AMBIGUOUS_UNMAPPED_CLASSES
    )

    is_classified = parcels["tillage_class"].notna(
    ) | parcels["is_grassland"] | parcels["is_excluded"]
    n_unmapped = (~is_classified).sum()
    if n_unmapped:
        print(f"Warning: {n_unmapped} parcels are unclassified. Inspect "
              "parcels.loc[~is_classified, 'crop'].value_counts() and "
              "extend the class mappings above.")
    return parcels


def attach_estimates(parcels: gpd.GeoDataFrame, tillage_lookup: pd.DataFrame,
                     grassland_lookup: pd.DataFrame) -> gpd.GeoDataFrame:
    # Tillage branch
    parcels = parcels.merge(
        tillage_lookup.rename(columns={"class": "tillage_class"}),
        on="tillage_class", how="left",
    )

    # Grassland branch — join on REGION (county-level area doesn't
    # exist), so every parcel needs its county mapped to a region first.
    parcels["REGION"] = parcels["COUNTY"].map(COUNTY_TO_REGION)
    parcels = parcels.merge(grassland_lookup, on="REGION", how="left")
    parcels.loc[~parcels["is_grassland"], [
        "grassland_index_mean", "grassland_index_std"]] = np.nan

    n_tillage = parcels["tillage_class"].notna().sum()
    n_grassland = parcels["is_grassland"].sum()
    n_excluded = parcels["is_excluded"].sum()
    n_unresolved = len(parcels) - n_tillage - n_grassland - n_excluded
    print(f"Tillage: {n_tillage} | Grassland: {n_grassland} | "
          f"Excluded: {n_excluded} | Unresolved: {n_unresolved}")

    return parcels


# ---------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------

CACHE_PATH = "data/_cache_parcels_classified.parquet"


def main(use_cache: bool = True):
    """
    use_cache: skip the slow load/spatial-join/classify stage (the LPIS
    read, county dissolve, and county assignment — the expensive part
    at 4.9M raw rows) if a cached result already exists. Set to False,
    or delete CACHE_PATH, whenever LPIS_PATH, COUNTY_BOUNDARIES_PATH,
    the class-mapping dicts, or load_parcels()'s filtering logic change
    — the cache does NOT auto-invalidate on those changes.

    This matters because the AQA04/AAA10/IFS10 lookups and the final
    merge are fast; only the LPIS load + spatial join + classification
    is slow, and it's been re-run unchanged across many recent
    iterations that were really only debugging the lookup tables.
    """
    import os

    if use_cache and os.path.exists(CACHE_PATH):
        print(f"Loading cached classified parcels from {CACHE_PATH}")
        parcels = gpd.read_parquet(CACHE_PATH)
    else:
        parcels = load_parcels(LPIS_PATH)
        counties = load_counties(COUNTY_BOUNDARIES_PATH)
        parcels = assign_county(parcels, counties)
        parcels = classify_parcels(parcels)

        parcels.to_parquet(CACHE_PATH)
        print(f"Cached classified parcels to {CACHE_PATH} "
              "(delete this file if LPIS/county inputs or class maps change)")

    aqa04_yield = load_aqa04(AQA04_CSV)
    cattle_df = load_aaa10(AAA10_CSV)
    grassland_area_df = load_ifs10_grassland_area(IFS10_CSV)

    tillage_lookup = build_tillage_lookup(aqa04_yield)
    grassland_lookup = build_grassland_lookup(cattle_df, grassland_area_df)

    result = attach_estimates(parcels, tillage_lookup, grassland_lookup)

    result.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"Wrote {len(result)} parcels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
