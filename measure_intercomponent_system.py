"""
Measures the distance BETWEEN connected components (not within them --
that's what the earlier nearest-neighbor check measured, which is why
it couldn't answer this question). Dissolves each component's parcels
into one polygon, then measures nearest-OTHER-component distance for
each. If these gaps are consistently small (a few meters) -- there's
still an undetected adjacency bug. If they're large and consistent
with real features (tens of meters+, i.e. roads/rivers/settlements) --
the fragmentation is likely a genuine landscape feature, not a bug.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from collections import deque

NODES_PATH = "data/graph_nodes.parquet"
EDGES_PATH = "data/graph_edges.parquet"
COMBINED_PATH = "data/parcel_yield_carbon.gpkg"
TARGET_COUNTY = "OFFALY"

nodes = pd.read_parquet(NODES_PATH)
edges = pd.read_parquet(EDGES_PATH)
county_mask = nodes["COUNTY"].str.upper() == TARGET_COUNTY.upper()
county_node_ids = set(nodes.index[county_mask])
in_county_edges = edges[edges["a"].isin(county_node_ids) & edges["b"].isin(county_node_ids)]

parent = {n: n for n in county_node_ids}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for a, b in zip(in_county_edges["a"], in_county_edges["b"]):
    union(a, b)

component_of = {n: find(n) for n in county_node_ids}

# Load real geometries for Offaly parcels, tag with component id, dissolve
gdf = gpd.read_file(COMBINED_PATH).reset_index(drop=True)
gdf["parcel_id"] = gdf.index
offaly = gdf[gdf["parcel_id"].isin(county_node_ids)][["parcel_id", "geometry"]].copy()
offaly["component"] = offaly["parcel_id"].map(component_of)

dissolved = offaly.dissolve(by="component").reset_index()
print(f"{len(dissolved)} components dissolved into single polygons each")

# Nearest OTHER component, for the largest few components specifically
# (most informative -- these are the ones that would anchor a case-study
# sub-region, so their isolation matters most)
dissolved["size"] = offaly.groupby("component").size().reindex(dissolved["component"]).values
top = dissolved.sort_values("size", ascending=False).head(10)

nearest = gpd.sjoin_nearest(top, dissolved, distance_col="dist", exclusive=True)
print("\nNearest-OTHER-component distance for the 10 largest components:")
print(nearest[["component_left", "size_left", "dist"]].to_string(index=False))
