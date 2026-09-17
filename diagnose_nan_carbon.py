"""
Quick diagnostic for parcel_ids [118, 130, 142] (and any others like
them) -- prints every field finalize_carbon_estimate() would have used
to decide their carbon value, to see WHY they fell through with NaN
instead of even the 'excluded' 0.0 default.
"""
import geopandas as gpd
from carbon_transition_delta import classify_land_cover

COMBINED_PATH = "data/parcel_yield_carbon.gpkg"
SUSPECT_PARCEL_IDS = [118, 130, 142]

gdf = gpd.read_file(COMBINED_PATH).reset_index(drop=True)
gdf["parcel_id"] = gdf.index
gdf["land_cover_class"] = gdf.apply(classify_land_cover, axis=1)

cols_to_check = [
    "parcel_id", "crop", "tillage_class", "is_grassland", "carbon_source",
    "is_excluded", "carbon_t_c_per_ha_mean", "land_cover_class",
    "Associat_S", "COUNTY",
]
available_cols = [c for c in cols_to_check if c in gdf.columns]

suspects = gdf[gdf["parcel_id"].isin(SUSPECT_PARCEL_IDS)]
print(suspects[available_cols].to_string(index=False))

# Also check: how many parcels NATIONALLY have this same NaN pattern,
# to see if it's 3 isolated cases or a systemic rate worth extrapolating
# to the full Offaly region before running the big search.
all_nan = gdf[gdf["carbon_t_c_per_ha_mean"].isna()]
print(f"\nTotal parcels nationally with NaN carbon_t_c_per_ha_mean: {len(all_nan):,} "
      f"({100*len(all_nan)/len(gdf):.3f}% of all {len(gdf):,} parcels)")
if len(all_nan) > 0:
    print("\nBreakdown by land_cover_class among the NaN parcels:")
    print(all_nan["land_cover_class"].value_counts().to_string())
    print("\nBreakdown by Associat_S (soil association) -- NaN counts, top 10:")
    if "Associat_S" in all_nan.columns:
        print(all_nan["Associat_S"].value_counts().head(10).to_string())
