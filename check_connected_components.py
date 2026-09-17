"""
Diagnoses whether Offaly's parcel adjacency graph has one dominant
giant component (in which case: just pick a seed from it) or is
genuinely fragmented into many small components (in which case:
build_parcel_graph.py's adjacency predicate likely needs a small
buffer/tolerance instead of exact geometric touching -- real LPIS
parcels often have small gaps between physically neighboring fields
that an exact-touch join would miss).
"""
import pandas as pd

NODES_PATH = "data/graph_nodes.parquet"
EDGES_PATH = "data/graph_edges.parquet"
TARGET_COUNTY = "OFFALY"

nodes = pd.read_parquet(NODES_PATH)
edges = pd.read_parquet(EDGES_PATH)

county_mask = nodes["COUNTY"].str.upper() == TARGET_COUNTY.upper()
county_node_ids = set(nodes.index[county_mask])
in_county_edges = edges[edges["a"].isin(county_node_ids) & edges["b"].isin(county_node_ids)]

# Union-find for connected components -- fast, no extra dependency needed
parent = {n: n for n in county_node_ids}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

for a, b in zip(in_county_edges["a"], in_county_edges["b"]):
    union(a, b)

component_sizes = {}
for n in county_node_ids:
    root = find(n)
    component_sizes[root] = component_sizes.get(root, 0) + 1

sizes = sorted(component_sizes.values(), reverse=True)
nodes_with_any_edge = set(in_county_edges["a"]) | set(in_county_edges["b"])
n_isolated = sum(1 for n in county_node_ids if n not in nodes_with_any_edge)

print(f"Total Offaly parcels: {len(county_node_ids):,}")
print(f"Total connected components: {len(sizes):,}")
print(f"Largest component: {sizes[0]:,} parcels ({100*sizes[0]/len(county_node_ids):.1f}% of county)")
print(f"Top 10 component sizes: {sizes[:10]}")
print(f"Parcels with ZERO edges at all (fully isolated nodes): {n_isolated:,}")
print(f"Components of size 1 (isolated after union-find): {sum(1 for s in sizes if s == 1):,}")
print(f"Components of size < 10: {sum(1 for s in sizes if s < 10):,}")
print(f"Components of size >= 500: {sum(1 for s in sizes if s >= 500):,}")
