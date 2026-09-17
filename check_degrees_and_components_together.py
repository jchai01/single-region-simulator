"""
Checks degree distribution AND connected components TOGETHER against
the CURRENT graph_edges.parquet (intersects, buffer=0), to settle two
open questions at once instead of ping-ponging:
  1. Is the degree-explosion (max 1405, 25,966 parcels >100) still
     present without buffering, or was that specific to the buffer?
  2. Given the small (1.7-5.75m) inter-component gaps just measured,
     is a modest evidence-based buffer (~6m) likely to close them
     safely, or would it reintroduce the explosion?
"""
import pandas as pd

EDGES_PATH = "data/graph_edges.parquet"

edges = pd.read_parquet(EDGES_PATH)
degree = pd.concat([edges["a"], edges["b"]]).value_counts()

print("=== Degree distribution (current: intersects, buffer=0) ===")
print(degree.describe())
print(f"\nMax degree: {degree.max()}")
print(f"Parcels with degree > 50: {(degree > 50).sum()}")
print(f"Parcels with degree > 100: {(degree > 100).sum()}")

worst_pid = degree.idxmax()
print(f"\nHighest-degree parcel_id: {worst_pid}, degree={degree.max()}")

# Also inspect what that parcel actually IS -- confirms or denies the
# "dense small-parcel cluster" hypothesis directly, rather than guessing
import geopandas as gpd
gdf = gpd.read_file("data/parcel_yield_carbon.gpkg").reset_index(drop=True)
gdf["parcel_id"] = gdf.index
worst_row = gdf[gdf["parcel_id"] == worst_pid]
print("\nHighest-degree parcel's own data:")
cols = [c for c in ["parcel_id", "crop", "COUNTY"] if c in worst_row.columns]
print(worst_row[cols].to_string(index=False))
print(f"Area: {worst_row.geometry.area.values[0] / 10_000:.4f} ha")
