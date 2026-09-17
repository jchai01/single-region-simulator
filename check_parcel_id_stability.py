"""
Decisive check: does parcel_id 118 mean the same physical parcel now
as it did earlier tonight? If parcel_yield_carbon.gpkg's row order has
shifted at any point, this will show a different COUNTY/crop/geometry
than the original diagnose_nan_carbon.py run did (LIMERICK, Permanent
Pasture, Tidal marsh).
"""
import geopandas as gpd

gdf = gpd.read_file("data/parcel_yield_carbon.gpkg").reset_index(drop=True)
gdf["parcel_id"] = gdf.index

for pid in [118, 130, 142]:
    row = gdf[gdf["parcel_id"] == pid].iloc[0]
    print(f"parcel_id {pid}: COUNTY={row.get('COUNTY')}, crop={row.get('crop')}, "
          f"Associat_S={row.get('Associat_S')}, "
          f"centroid={row.geometry.centroid.x:.0f},{row.geometry.centroid.y:.0f}")

print(f"\nTotal row count in file: {len(gdf):,}")
print("If COUNTY here is OFFALY instead of LIMERICK for parcel_id 118, "
      "that CONFIRMS row order has shifted since the original diagnose_nan_carbon.py run.")
