"""
Pair yield_model_v1.py and carbon_model_v1.py outputs into one parcel
table, for looking at the yield/carbon tradeoff the optimizer will
eventually need to reason about.

Join strategy: both outputs derive from the SAME cached, classified
parcel set (data/_cache_parcels_classified.parquet), processed with
only left-merges that preserve row order and count (both confirmed at
1,541,057 rows). Rather than relying on implicit row-index alignment
(fragile -- breaks silently if either pipeline changes), this joins on
each parcel's exact geometry (as WKB bytes), which is a genuine,
stable identity key since geometry is untouched by either pipeline
after the shared cache is built.

NOTE: this is a stopgap. For robustness going into the generative/
optimizer stage, consider adding a persistent parcel_id column at
cache-build time (in yield_model_v1.py's load_parcels()) instead of
relying on geometry-as-key indefinitely.
"""

import geopandas as gpd
import pandas as pd

YIELD_PATH = "data/parcel_yield_lookup.gpkg"
CARBON_PATH = "data/parcel_carbon_lookup.gpkg"
OUTPUT_PATH = "data/parcel_yield_carbon.gpkg"

# Columns from the carbon output that are genuinely NEW information --
# everything else (crop, COUNTY, REGION, is_grassland, is_excluded,
# tillage_class) is shared/duplicated from the same source cache and
# should come from the yield table only, to avoid redundant columns.
CARBON_NEW_COLUMNS = [
    "carbon_t_c_per_ha_mean", "carbon_t_c_per_ha_std", "carbon_source",
    "Associat_S", "n_series_used",
]


def main():
    yield_gdf = gpd.read_file(YIELD_PATH)
    carbon_gdf = gpd.read_file(CARBON_PATH)

    print(f"Yield table: {len(yield_gdf)} rows")
    print(f"Carbon table: {len(carbon_gdf)} rows")
    if len(yield_gdf) != len(carbon_gdf):
        raise ValueError(
            "Row counts differ -- positional join below is NOT safe. "
            "Investigate before proceeding (do not fall back to a "
            "geometry/WKB join -- confirmed to explode on duplicate "
            "geometries and exhaust disk space)."
        )

    # Positional join, not geometry-content join. Both outputs derive
    # from the same cache via only left-merges (which preserve row
    # order in pandas), and row counts match exactly -- confirmed safe.
    # A WKB/geometry-content join was tried first and BLEW UP (millions
    # of rows from a many-to-many match) because some parcels share
    # identical or near-identical geometry -- duplicate keys make a
    # content-based join unsafe here. Positional join avoids that
    # entirely, at the cost of being fragile if either pipeline is
    # later changed to reorder or filter rows differently -- add a
    # real parcel_id column at the next full cache rebuild to remove
    # this fragility permanently.
    yield_gdf = yield_gdf.reset_index(drop=True)
    carbon_gdf = carbon_gdf.reset_index(drop=True)

    carbon_subset = carbon_gdf[CARBON_NEW_COLUMNS]
    combined = pd.concat([yield_gdf, carbon_subset], axis=1)

    n_unmatched = combined["carbon_t_c_per_ha_mean"].isna().sum()
    print(f"Parcels with no carbon value: {n_unmatched}")
    combined.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"Wrote {len(combined)} combined parcels to {OUTPUT_PATH}")

    # --- Summary: yield vs carbon by branch, for a first look at the tradeoff ---
    print("\n--- Tillage: yield (t/ha) vs carbon (t C/ha) ---")
    tillage = combined[combined["tillage_class"].notna()]
    print(tillage[["yield_mean", "carbon_t_c_per_ha_mean"]].describe())

    print("\n--- Grassland: productivity index vs carbon (t C/ha) ---")
    grass = combined[combined["is_grassland"]]
    print(grass[["grassland_index_mean", "carbon_t_c_per_ha_mean"]].describe())

    print("\n--- Forestry & peatland: carbon only (no yield reward defined for these) ---")
    forest_peat = combined[combined["carbon_source"].isin(
        ["nfi_blended_average", "peatland_literature"])]
    print(forest_peat.groupby("carbon_source")[
          "carbon_t_c_per_ha_mean"].describe())

    print("\n--- Overall mean carbon by land-cover branch ---")
    print(combined.groupby("carbon_source")[
          "carbon_t_c_per_ha_mean"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
