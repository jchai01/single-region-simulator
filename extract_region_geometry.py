"""
One-time extraction: pulls just the parcels that actually appear in
your scenario exports out of the full national parcel_yield_carbon.gpkg
(~1.54M parcels), and saves a small region-only geometry file for the
app to use instead.

THIS IS THE FIX for the app hanging/crashing, especially in split view:
the app was loading and merging against the FULL national file every
time, and -- worse -- rendering every parcel NOT in your region as a
"not reallocatable" layer covering nearly the whole country. This cuts
that down from ~1.54M polygons to a few hundred, a >1000x reduction.

Run this ONCE (or again whenever your case-study region changes).
After this, point the app at data/region_geometry.gpkg instead of the
full national file -- already done in the updated app.py below.
"""
import glob
import os

import geopandas as gpd
import pandas as pd

FULL_GEOMETRY_PATH = "data/parcel_yield_carbon.gpkg"
OUTPUT_PATH = "data/region_geometry.gpkg"


def get_needed_parcel_ids():
    """Union of every parcel_id appearing in any scenario export --
    exactly what the app can ever need to render, nothing more."""
    ids = set()
    for f in glob.glob("data/scenario_details_seed*.parquet"):
        ids.update(pd.read_parquet(f, columns=["parcel_id"])["parcel_id"].unique())
    if os.path.exists("data/scenario_details_baselines.parquet"):
        ids.update(
            pd.read_parquet("data/scenario_details_baselines.parquet",
                            columns=["parcel_id"])["parcel_id"].unique()
        )
    return ids


if __name__ == "__main__":
    needed_ids = get_needed_parcel_ids()
    if not needed_ids:
        raise SystemExit(
            "No scenario export files found in data/ -- run your NSGA-II "
            "seeds and baseline_comparison.py first, then run this."
        )
    print(f"Found {len(needed_ids)} unique parcel_ids referenced across all scenario exports")

    print(f"Reading {FULL_GEOMETRY_PATH} -- this is the slow part, and the ONLY "
          "time this large file needs to be touched.")
    gdf = gpd.read_file(FULL_GEOMETRY_PATH)
    if "parcel_id" not in gdf.columns:
        gdf["parcel_id"] = gdf.index
    print(f"Full national file: {len(gdf):,} parcels")

    region = gdf[gdf["parcel_id"].isin(needed_ids)][["parcel_id", "geometry"]].copy()
    print(f"Filtered down to {len(region)} parcels -- this is what the app will actually use")

    region.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"\nSaved to {OUTPUT_PATH}. The app is already updated to use this file.")
