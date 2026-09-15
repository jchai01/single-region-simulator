"""
Parcel adjacency graph construction for LandFuture DSS's generative model.

Builds a graph where:
  - NODES = individual parcels (from the paired yield/carbon dataset),
    carrying features: land_cover_class, yield/carbon estimates,
    soil_cluster/Associat_S, COUNTY/REGION.
  - EDGES = parcels that share a physical border (queen contiguity:
    shared edge OR shared vertex -- standard choice for land-use
    adjacency; rook contiguity, edge-only, is stricter and would miss
    diagonally-touching parcels).

This is the shared input both candidate architectures (masked-node GNN
v1, graph diffusion v2 if needed later) will consume -- built once,
independent of which model trains on it.

SCALE WARNING: exact polygon-adjacency detection (checking every pair
of parcels for a shared border) is O(n^2) if done naively -- infeasible
at 1.5M parcels. This script uses a spatial index (via geopandas'
sjoin, which uses an internal STRtree) to only compare parcels whose
bounding boxes are near each other, not all pairs -- but even so, this
is the heaviest single step in the pipeline so far. Expect it to take
a while; the output is cached so this cost is paid once.
"""

import geopandas as gpd
import pandas as pd
import numpy as np

INPUT_PATH = "data/parcel_yield_carbon.gpkg"
GRAPH_NODES_PATH = "data/graph_nodes.parquet"
GRAPH_EDGES_PATH = "data/graph_edges.parquet"

# Columns to retain as node features -- keep this list deliberate and
# small rather than carrying every column through; the GNN only needs
# what it will actually condition on / predict.
NODE_FEATURE_COLUMNS = [
    "crop", "tillage_class", "is_grassland", "is_excluded",
    "COUNTY", "REGION", "Associat_S",
    "yield_mean", "yield_std",
    "grassland_index_mean", "grassland_index_std",
    "carbon_t_c_per_ha_mean", "carbon_t_c_per_ha_std", "carbon_source",
]


def load_parcels(path: str = INPUT_PATH) -> gpd.GeoDataFrame:
    parcels = gpd.read_file(path)
    parcels = parcels.reset_index(drop=True)
    parcels["parcel_id"] = parcels.index  # first real persistent ID in this pipeline
    print(f"Loaded {len(parcels)} parcels, assigned parcel_id 0..{len(parcels)-1}")
    return parcels


def build_adjacency_edges(parcels: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Queen contiguity: two parcels are adjacent if their geometries
    touch at all (shared edge or shared vertex). Uses geopandas' sjoin
    with predicate='touches', which internally uses a spatial index
    (STRtree) to avoid all-pairs comparison.

    Returns a DataFrame of (parcel_id_a, parcel_id_b) pairs, each
    unordered pair listed once (a < b) -- the GNN/diffusion model can
    treat this as an undirected graph and symmetrize as needed at
    training time.
    """
    minimal = parcels[["parcel_id", "geometry"]].copy()

    joined = gpd.sjoin(minimal, minimal, how="inner", predicate="touches")
    joined = joined.rename(columns={"parcel_id_left": "a", "parcel_id_right": "b"})

    # Dedup: sjoin with predicate='touches' returns both (a,b) and
    # (b,a) since it's a self-join -- keep only a < b.
    edges = joined[joined["a"] < joined["b"]][["a", "b"]].drop_duplicates()

    print(f"Found {len(edges)} adjacency edges for {len(parcels)} parcels "
          f"(avg degree: {2 * len(edges) / len(parcels):.1f})")
    return edges.reset_index(drop=True)


def build_node_table(parcels: gpd.GeoDataFrame) -> pd.DataFrame:
    """Node feature table, keyed by parcel_id. Geometry dropped here --
    kept in a separate lightweight table if needed for visualization,
    since the GNN itself trains on features/graph structure, not raw
    coordinates."""
    cols = ["parcel_id"] + [c for c in NODE_FEATURE_COLUMNS if c in parcels.columns]
    missing = [c for c in NODE_FEATURE_COLUMNS if c not in parcels.columns]
    if missing:
        print(f"Warning: expected node feature columns not found, skipped: {missing}")
    return parcels[cols].copy()


def main():
    parcels = load_parcels()
    nodes = build_node_table(parcels)
    edges = build_adjacency_edges(parcels)

    nodes.to_parquet(GRAPH_NODES_PATH)
    edges.to_parquet(GRAPH_EDGES_PATH)
    print(f"Wrote {len(nodes)} nodes to {GRAPH_NODES_PATH}")
    print(f"Wrote {len(edges)} edges to {GRAPH_EDGES_PATH}")

    # Basic connectivity sanity check: how many parcels have ZERO
    # neighbors? These would be isolated nodes in the graph -- worth
    # knowing before training (could indicate real geographic
    # isolation, e.g. an island parcel, or a geometry/topology issue).
    connected_ids = set(edges["a"]) | set(edges["b"])
    n_isolated = len(nodes) - len(connected_ids & set(nodes["parcel_id"]))
    print(f"Isolated parcels (no detected neighbors): {n_isolated} "
          f"({n_isolated / len(nodes):.1%})")


if __name__ == "__main__":
    main()
