"""
Does the national NaN-carbon pattern (Rock/Island/Tidal marsh/etc. soil
associations, 2.5% nationally) actually matter for the Offaly case
study specifically? National rate doesn't tell you this directly --
these substrates are geographically clustered (coastal, upland), and
Offaly is neither.
"""
import geopandas as gpd
from carbon_transition_delta import classify_land_cover

COMBINED_PATH = "data/parcel_yield_carbon.gpkg"

gdf = gpd.read_file(COMBINED_PATH).reset_index(drop=True)
gdf["parcel_id"] = gdf.index
gdf["land_cover_class"] = gdf.apply(classify_land_cover, axis=1)

offaly = gdf[gdf["COUNTY"].str.upper() == "OFFALY"]
offaly_nan = offaly[offaly["carbon_t_c_per_ha_mean"].isna()]

print(f"Offaly: {len(offaly):,} parcels total, {len(offaly_nan):,} with NaN carbon "
      f"({100*len(offaly_nan)/len(offaly):.3f}%)")
if len(offaly_nan) > 0:
    print("\nOffaly's NaN parcels by Associat_S:")
    print(offaly_nan["Associat_S"].value_counts().to_string())
    print("\nOffaly's NaN parcels by land_cover_class:")
    print(offaly_nan["land_cover_class"].value_counts().to_string())
