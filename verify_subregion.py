"""
Self-contained verification of the selected Offaly sub-region -- loads
parcel_ids from the saved file rather than assuming they're already in
scope, so this can be run standalone.
"""
import geopandas as gpd
from carbon_transition_delta import classify_land_cover

with open("data/offaly_subregion_parcel_ids.txt") as f:
    parcel_ids = [int(line.strip()) for line in f if line.strip()]

print(f"Loaded {len(parcel_ids)} parcel_ids from file")

combined = gpd.read_file("data/parcel_yield_carbon.gpkg").reset_index(drop=True)
combined["parcel_id"] = combined.index
combined["land_cover_class"] = combined.apply(classify_land_cover, axis=1)

region = combined[combined["parcel_id"].isin(parcel_ids)]
print(f"\nMatched {len(region)} of {len(parcel_ids)} requested parcel_ids in combined data")

print("\nCOUNTY breakdown (should show ~100% OFFALY -- confirms node/parcel_id correspondence):")
print(region["COUNTY"].value_counts().to_string())

print("\nland_cover_class composition:")
print(region["land_cover_class"].value_counts().to_string())
