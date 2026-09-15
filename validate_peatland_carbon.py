"""
Polygon-intersection validation of LandFuture DSS peatland carbon estimates
against Habib & Connolly (2023) land-use-change dataset.

Replaces the earlier centroid-based area comparison. Instead of asking
"does this parcel's centroid fall inside a peatland pixel?", this computes
actual area-of-overlap between LPIS BOG parcels and the Habib & Connolly
polygonized raster, then turns that into carbon-weighted agreement/
disagreement rather than just hectare counts.

Data:
  - LPIS 2025 parcels, filtered to `crop == 'BOG'` (vector, assumed EPSG:2157 / ITM)
  - Habib & Connolly (2023) OSF deposit (10.17605/OSF.IO/V4NSQ): three rasters,
    one per epoch (1989-91, 2004-06, 2018-20). Use the 2018-2020 raster as the
    closest temporal match to LPIS 2025. CRS unconfirmed on your download --
    check and reproject; Irish Landsat/GEE products are commonly delivered in
    EPSG:32629 (UTM 29N) rather than ITM.

Fill in the CONFIG block below with your actual paths, band, and class codes,
then run the three stages (polygonize -> overlay -> carbon-weighted metrics)
in order. Each stage writes an intermediate file so you can inspect output
before moving to the next (e.g. open the polygonized raster in QGIS to sanity
check it before running the overlay).
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# CONFIG -- edit these
# ---------------------------------------------------------------------------

HABIB_CONNOLLY_RASTER = "./data/Peatlands_Landuse_Rasters_Ireland/Landuse_2020.tif"
# Class code(s) in the raster that represent the peatland classes you want
# to validate against. Check the deposit's metadata/README for the class
# legend -- Habib & Connolly's later LUCIP scheme (7-class) is documented in
# their 2024 Sci Reports follow-up; the 2023 OSF raster's own legend may
# differ, confirm from the deposit itself rather than assuming it matches.
# confirmed via magnitude match against Habib & Connolly (2023) Table 5:
PEATLAND_CLASS_CODES = [4]
# code 1=Industrial (41.7k ha), 2=Forest (224.7k ha), 3=Grassland (357.9k ha), 4=Residual peatland (835.7k ha).
# Residual peatland (4) is the class comparable to LPIS's declared BOG -- not Industrial (1),
# which is BnM-managed extraction sites and would never appear as farmed/declared agricultural land.

# carbon_model_v1.py's OUTPUT_PATH -- preferred over the raw cache parquet
LPIS_SOURCE = "data/parcel_carbon_lookup.gpkg"
# ITM -- confirm this matches what your LPIS/carbon model already uses
TARGET_CRS = "EPSG:2157"

# Per-hectare carbon constants already established in the carbon model
CARBON_CONSTANTS_T_PER_HA = {
    "blanket_bog": 705,       # +/- 150, western blanket peat 250-1000 range
    "raised_bog": 1160,       # upper bound; regional variation applies
}
# Which constant to apply to LPIS BOG parcels for the carbon-weighted metric.
# If your BOG class doesn't distinguish blanket/raised, you likely need a
# spatial join against a bog-type layer (e.g. DIPMv2 or NPWS peatland types)
# before this step -- flagging rather than assuming, since this materially
# changes the carbon-weighted result.
DEFAULT_BOG_CARBON_T_PER_HA = CARBON_CONSTANTS_T_PER_HA["blanket_bog"]

OUT_POLYGONIZED = "./data/habib_connolly_peatland_polygons.gpkg"
OUT_OVERLAY = "./data/lpis_bog_vs_habib_connolly_overlay.gpkg"
OUT_METRICS = "./data/validation_metrics.csv"


# ---------------------------------------------------------------------------
# Stage 1: polygonize the Habib & Connolly raster
# ---------------------------------------------------------------------------

def polygonize_peatland_raster(raster_path, class_codes, out_path):
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        transform = src.transform
        raster_crs = src.crs
        nodata = src.nodata

    mask = np.isin(band, class_codes)
    if nodata is not None:
        mask &= band != nodata

    if not mask.any():
        raise ValueError(
            f"No pixels matched class codes {class_codes}. "
            f"Unique values in raster: {np.unique(band)[:20]} ... "
            "confirm PEATLAND_CLASS_CODES against the deposit's legend."
        )

    geoms = [
        {"geometry": shape(geom), "class": val}
        for geom, val in shapes(band, mask=mask, transform=transform)
    ]
    gdf = gpd.GeoDataFrame(geoms, crs=raster_crs)

    # Landsat-derived polygons are blocky (30m pixels) -- fix any
    # self-intersections from the polygonize step before doing anything else
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]

    # Dissolve adjacent same-class pixels into single polygons per contiguous area
    gdf = gdf.dissolve(by="class").reset_index()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    gdf = gdf.to_crs(TARGET_CRS)
    gdf["area_ha"] = gdf.geometry.area / 10_000

    gdf.to_file(out_path, driver="GPKG")
    print(f"Polygonized: {len(gdf)} peatland polygons, "
          f"{gdf['area_ha'].sum():,.1f} ha total, CRS reprojected to {TARGET_CRS}")
    return gdf


# ---------------------------------------------------------------------------
# Stage 2: overlay against LPIS BOG parcels
# ---------------------------------------------------------------------------

def load_lpis_bog(lpis_path, source_crs=None):
    """
    Load parcels and filter to BOG, from either:
      - a GeoPackage/Shapefile/etc (gpd.read_file) -- CRS is embedded,
        source_crs is ignored
      - a parquet file -- tries GeoParquet first, falls back to plain
        pandas + manual geometry reconstruction if `geo` metadata is
        missing, in which case source_crs MUST be supplied (a non-
        GeoParquet file carries no embedded CRS to recover)

    Recommended source: carbon_model_v1.py's own OUTPUT_PATH
    (data/parcel_carbon_lookup.gpkg) -- the finalized output, already
    has `crop` + geometry + carbon estimates merged, and GPKG sidesteps
    the GeoParquet-metadata failure mode entirely.
    """
    if str(lpis_path).lower().endswith((".gpkg", ".shp", ".geojson")):
        parcels = gpd.read_file(lpis_path)
        bog = parcels[parcels["crop"].str.upper() == "BOG"].copy()
        return bog

    try:
        lpis = gpd.read_parquet(lpis_path, columns=["crop", "geometry"])
        bog = lpis[lpis["crop"].str.upper() == "BOG"].copy()
        if bog.crs is None:
            bog = bog.set_crs(source_crs)
        return bog
    except ValueError as e:
        if "Missing geo metadata" not in str(e):
            raise
        print("No GeoParquet metadata found -- falling back to plain pandas "
              "read + manual geometry reconstruction.")

    df = pd.read_parquet(lpis_path, columns=["crop", "geometry"])
    df = df[df["crop"].str.upper() == "BOG"].copy()

    sample = df["geometry"].iloc[0]
    if isinstance(sample, (bytes, bytearray)):
        geom = gpd.GeoSeries.from_wkb(df["geometry"])
    elif isinstance(sample, str):
        geom = gpd.GeoSeries.from_wkt(df["geometry"])
    else:
        geom = gpd.GeoSeries(df["geometry"])

    bog = gpd.GeoDataFrame(
        df.drop(columns=["geometry"]), geometry=geom, crs=source_crs)
    return bog


def overlay_bog_parcels(lpis_path, habib_connolly_gdf, out_path, lpis_source_crs=TARGET_CRS):
    bog = load_lpis_bog(lpis_path, lpis_source_crs)
    bog = bog.to_crs(TARGET_CRS)
    bog["lpis_area_ha"] = bog.geometry.area / 10_000
    bog = bog[bog.geometry.is_valid]

    hc = habib_connolly_gdf.to_crs(
        TARGET_CRS) if habib_connolly_gdf.crs != TARGET_CRS else habib_connolly_gdf

    bog_r = bog.reset_index().rename(columns={"index": "lpis_id"})
    hc_r = hc.reset_index().rename(columns={"index": "hc_id"})

    # `how="intersection"` (not "union") -- uses the spatial index to only
    # process pairs whose bounding boxes actually overlap, instead of
    # computing the full planar partition of both datasets. For ~40k x 15k
    # polygons nationally, "union" was almost certainly the hang.
    print("Computing agreement (intersection)...")
    agreement = gpd.overlay(
        bog_r, hc_r, how="intersection", keep_geom_type=True)
    agreement["overlap_area_ha"] = agreement.geometry.area / 10_000
    agreement = agreement[agreement["overlap_area_ha"] > 0.0001]
    print(f"  {len(agreement)} agreement regions")

    print("Computing LPIS-only (difference)...")
    lpis_only = gpd.overlay(bog_r, hc_r, how="difference", keep_geom_type=True)
    lpis_only = lpis_only.explode(index_parts=False).reset_index(drop=True)
    lpis_only["overlap_area_ha"] = lpis_only.geometry.area / 10_000
    lpis_only = lpis_only[lpis_only["overlap_area_ha"] > 0.0001]
    print(f"  {len(lpis_only)} LPIS-only regions")

    print("Computing Habib & Connolly-only (difference)...")
    hc_only = gpd.overlay(hc_r, bog_r, how="difference", keep_geom_type=True)
    hc_only = hc_only.explode(index_parts=False).reset_index(drop=True)
    hc_only["overlap_area_ha"] = hc_only.geometry.area / 10_000
    hc_only = hc_only[hc_only["overlap_area_ha"] > 0.0001]
    print(f"  {len(hc_only)} Habib & Connolly-only regions")

    agreement["hc_id"] = agreement.get("hc_id")
    agreement["lpis_id"] = agreement.get("lpis_id")
    lpis_only["hc_id"] = None
    hc_only["lpis_id"] = None

    overlay = pd.concat(
        [agreement[["lpis_id", "hc_id", "overlap_area_ha", "geometry"]],
         lpis_only[["lpis_id", "hc_id", "overlap_area_ha", "geometry"]],
         hc_only[["lpis_id", "hc_id", "overlap_area_ha", "geometry"]]],
        ignore_index=True,
    )
    overlay = gpd.GeoDataFrame(overlay, crs=TARGET_CRS)

    overlay.to_file(out_path, driver="GPKG")
    print(f"Overlay complete: {len(overlay)} regions total")
    return overlay, bog, hc


# ---------------------------------------------------------------------------
# Stage 3: area-weighted and carbon-weighted agreement metrics
# ---------------------------------------------------------------------------

def compute_metrics(overlay, bog, hc, out_path):
    # in LPIS BOG, not in H&C peatland
    lpis_only = overlay[overlay["hc_id"].isna()]
    # in H&C peatland, not in LPIS BOG
    hc_only = overlay[overlay["lpis_id"].isna()]
    agreement = overlay[overlay["hc_id"].notna() & overlay["lpis_id"].notna()]

    agreement_ha = agreement["overlap_area_ha"].sum()
    lpis_only_ha = lpis_only["overlap_area_ha"].sum()
    hc_only_ha = hc_only["overlap_area_ha"].sum()

    total_lpis_bog_ha = bog["lpis_area_ha"].sum()
    total_hc_peat_ha = hc["area_ha"].sum()
    union_ha = agreement_ha + lpis_only_ha + hc_only_ha

    iou = agreement_ha / union_ha if union_ha else float("nan")
    dice = 2 * agreement_ha / (total_lpis_bog_ha + total_hc_peat_ha) if (
        total_lpis_bog_ha + total_hc_peat_ha) else float("nan")

    # Carbon-weighted view: convert area agreement/disagreement into tonnes C,
    # so the headline validation number is defensible in carbon terms, not
    # just land-cover-class terms.
    agreement_tC = agreement_ha * DEFAULT_BOG_CARBON_T_PER_HA
    lpis_only_tC = lpis_only_ha * DEFAULT_BOG_CARBON_T_PER_HA
    hc_only_tC = hc_only_ha * DEFAULT_BOG_CARBON_T_PER_HA   # carbon your model MISSES

    metrics = pd.DataFrame([{
        "total_lpis_bog_ha": total_lpis_bog_ha,
        "total_habib_connolly_peatland_ha": total_hc_peat_ha,
        "agreement_ha": agreement_ha,
        "lpis_only_ha_overclaim": lpis_only_ha,
        "habib_connolly_only_ha_missed": hc_only_ha,
        "IoU": iou,
        "dice_coefficient": dice,
        "agreement_tC": agreement_tC,
        "overclaim_tC": lpis_only_tC,
        "missed_tC": hc_only_tC,
        "carbon_constant_used_t_per_ha": DEFAULT_BOG_CARBON_T_PER_HA,
    }])
    metrics.to_csv(out_path, index=False)
    print(metrics.to_string(index=False))
    print(f"\nNote: overclaim/missed tonnes use a single blanket-bog constant "
          f"({DEFAULT_BOG_CARBON_T_PER_HA} t/ha) applied uniformly. If blanket "
          f"vs raised bog isn't distinguished in your BOG class, this "
          f"understates uncertainty -- treat as a first-pass figure.")
    return metrics


if __name__ == "__main__":
    hc_gdf = polygonize_peatland_raster(
        HABIB_CONNOLLY_RASTER, PEATLAND_CLASS_CODES, OUT_POLYGONIZED
    )
    overlay, bog, hc = overlay_bog_parcels(LPIS_SOURCE, hc_gdf, OUT_OVERLAY)
    compute_metrics(overlay, bog, hc, OUT_METRICS)
