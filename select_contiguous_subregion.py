"""
Selects a genuinely contiguous sub-region within a county, using the
parcel adjacency graph (graph_nodes.parquet / graph_edges.parquet from
build_parcel_graph.py) -- NOT an arbitrary parcel_id slice. Contiguity
matters here because the whole reallocation-realism story (and the
GNN's own training objective) depends on spatially coherent regions,
not a scattered sample that happens to share a county label.

METHOD: BFS from a seed parcel, restricted to edges where BOTH
endpoints are in the target county, growing outward until the target
parcel count is reached. This guarantees every included parcel is
graph-connected to the seed via a chain of shared-boundary neighbors --
a genuine contiguous block, not just "nearby by chance."

Node/parcel_id correspondence: graph_nodes.parquet's row position is
assumed to match generate_scenarios.py's parcel_id = combined.index
convention (both described as row-order-preserving from the same
pipeline) -- if this assumption is wrong for your actual files, the
selected "parcel_ids" here won't line up with combined's rows, so
CHECK THIS FIRST via the verification block at the bottom before
trusting the output for a real run.
"""

import pandas as pd
import numpy as np
from collections import deque

NODES_PATH = "data/graph_nodes.parquet"
EDGES_PATH = "data/graph_edges.parquet"

TARGET_COUNTY = "OFFALY"
TARGET_PARCEL_COUNT = 750   # mid-point of the 500-1,000 range discussed
SEED_STRATEGY = "random"    # or set a specific node index directly


def select_contiguous_subregion(target_county, target_count, seed=0):
    nodes = pd.read_parquet(NODES_PATH)
    edges = pd.read_parquet(EDGES_PATH)

    county_mask = nodes["COUNTY"].str.upper() == target_county.upper()
    county_node_ids = set(nodes.index[county_mask])
    print(f"{target_county}: {len(county_node_ids):,} total parcels in the graph")

    # Restrict edges to those with BOTH endpoints inside the county --
    # cross-county edges are dropped so BFS can't "leak" across the
    # county boundary into a neighboring county's parcels.
    in_county_edges = edges[
        edges["a"].isin(county_node_ids) & edges["b"].isin(county_node_ids)
    ]
    print(f"{len(in_county_edges):,} edges fully within {target_county}")

    adjacency = {}
    for a, b in zip(in_county_edges["a"], in_county_edges["b"]):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    rng = np.random.default_rng(seed)
    seed_node = rng.choice(list(county_node_ids)) if SEED_STRATEGY == "random" else seed

    visited = {seed_node}
    queue = deque([seed_node])
    while queue and len(visited) < target_count:
        current = queue.popleft()
        for neighbor in adjacency.get(current, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= target_count:
                    break

    print(f"Selected {len(visited)} contiguous parcels (target was {target_count})")
    if len(visited) < target_count:
        print(f"WARNING: ran out of connected parcels before reaching the target -- "
              f"the seed's connected component has only {len(visited)} parcels. "
              "Try a different seed, or this county's graph may be fragmented "
              "into disconnected components (check for isolated parcels).")

    return sorted(visited)


if __name__ == "__main__":
    parcel_ids = select_contiguous_subregion(TARGET_COUNTY, TARGET_PARCEL_COUNT)

    # Save for reuse across benchmark/search/baseline scripts without
    # re-running BFS each time.
    with open("data/offaly_subregion_parcel_ids.txt", "w") as f:
        f.write("\n".join(str(p) for p in parcel_ids))
    print(f"Saved {len(parcel_ids)} parcel_ids to data/offaly_subregion_parcel_ids.txt")

    # --- Sanity checks before trusting this for a real run ---
    print("\n--- Verify node/parcel_id correspondence before using this region ---")
    print("Run this against your actual combined data:")
    print("""
import geopandas as gpd
combined = gpd.read_file("data/parcel_yield_carbon.gpkg").reset_index(drop=True)
combined["parcel_id"] = combined.index
region = combined[combined["parcel_id"].isin(parcel_ids)]
print(region["COUNTY"].value_counts())  # should show ~100% OFFALY
print(region["land_cover_class"].value_counts())  # composition sanity check
""")
