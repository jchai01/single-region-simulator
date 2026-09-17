"""
Measures the ACTUAL nearest-neighbor gap distances between Offaly
parcels, instead of guessing a buffer_distance by trial and error.
Uses gpd.sjoin_nearest, which returns real distances -- this tells us
whether the gaps are mostly small digitization slivers (a buffer of a
few meters would fix it) or something structurally different (real
physical features -- roads, hedgerows, ditches -- which a buffer
large enough to bridge would also risk false-connecting parcels
across, e.g., a road).
"""
import geopandas as gpd
import numpy as np

COMBINED_PATH = "data/parcel_yield_carbon.gpkg"
TARGET_COUNTY = "OFFALY"

gdf = gpd.read_file(COMBINED_PATH).reset_index(drop=True)
gdf["parcel_id"] = gdf.index
offaly = gdf[gdf["COUNTY"].str.upper() == TARGET_COUNTY.upper()][["parcel_id", "geometry"]].copy()

print(f"CRS: {offaly.crs} (confirm this is projected/meters before trusting distances below)")
print(f"{len(offaly)} Offaly parcels")

# Nearest OTHER parcel for each parcel (distance_col gives real distance
# in the CRS's units -- meters, if ITM/EPSG:2157 as expected)
nearest = gpd.sjoin_nearest(offaly, offaly, distance_col="dist", exclusive=True)
# exclusive=True (geopandas >=0.14) excludes self-matches; if your
# version doesn't support it, filter out dist==0 rows manually instead.

distances = nearest["dist"]
print(f"\nNearest-neighbor gap distance distribution (meters):")
for pct in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{pct}: {np.percentile(distances, pct):.2f}m")
print(f"  max: {distances.max():.2f}m")
print(f"  parcels with nearest neighbor > 10m away: {(distances > 10).sum()} "
      f"({100*(distances > 10).mean():.1f}%)")
print(f"  parcels with nearest neighbor > 20m away: {(distances > 20).sum()} "
      f"({100*(distances > 20).mean():.1f}%)")
