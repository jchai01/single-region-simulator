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


def build_adjacency_edges(parcels: gpd.GeoDataFrame, buffer_distance: float = 6.0) -> pd.DataFrame:
    """
    Queen contiguity: two parcels are adjacent if their geometries touch
    OR overlap at all.

    HISTORY OF THIS FUNCTION'S BUGS, both confirmed empirically before
    fixing (see the project's own diagnostic scripts):

    1. ORIGINAL VERSION used predicate='touches', which requires
       boundary-only contact with ZERO shared interior area -- it
       returns False for two polygons that overlap even by a sliver.
       Measuring actual nearest-neighbor distances on a real county
       showed >95% of parcels already sit at EXACTLY 0m from their
       nearest neighbor -- i.e. they're touching or overlapping, not
       gapped -- yet the graph was still catastrophically fragmented
       (3,733 components for one county, largest only 207 parcels).
       That combination (0m distances, still fragmented) means the
       real problem was OVERLAPPING geometries (a common real-world
       digitization artifact) being silently rejected by touches(),
       not gaps.

    2. FIRST ATTEMPTED FIX buffered every parcel by 1m and switched to
       predicate='intersects'. This was solving the WRONG problem
       (gaps) for the actual root cause (overlaps), and uniformly
       expanding every parcel's footprint by 1m created a false-
       adjacency explosion in any densely packed cluster (small
       buildings, subdivided plots): degree distribution afterward had
       a reasonable median (3) but a catastrophic tail (max 1,405
       "neighbors", 25,966 parcels with degree > 100) -- no real field
       touches over 100 others.

    3. FIX FOR OVERLAP-REJECTION: predicate='intersects' on the ORIGINAL,
       UNBUFFERED geometries. This correctly catches both clean touches
       and overlapping-sliver cases without artificially expanding
       anyone's footprint, so it doesn't create the dense-cluster
       explosion. Re-checked the degree distribution afterward (not just
       connectivity) specifically to rule out that concern -- confirmed
       the earlier high-degree tail (max ~1,404) is UNCHANGED with or
       without buffering, so it's real (a large Mayo commonage parcel
       genuinely bordering hundreds of smallholdings, confirmed via
       direct geometry inspection: valid single polygon, plausible
       perimeter/area ratio), not a buffering artifact.

    4. REMAINING GAP, MEASURED NOT GUESSED: after fix #3, the 10 largest
       connected components in a real county were still separated from
       their nearest other component by small, consistent gaps (1.7m to
       5.75m, measured via sjoin_nearest on dissolved per-component
       geometries) -- too narrow and too consistent to be roads/rivers,
       plausibly a hedgerow-and-ditch field boundary. buffer_distance
       default set to 6.0m (a small margin above the largest gap
       actually measured, 5.75m) to close this specific, evidenced gap
       -- NOT a guessed round number. If you change this, re-measure the
       gap distribution first (see measure_intercomponent_distances.py)
       rather than picking a new value by trial and error, and re-check
       the degree distribution afterward (see
       check_degree_and_components_together.py) to confirm this buffer
       size doesn't reintroduce the dense-cluster explosion pattern from
       fix attempt #2 -- 6m is small enough that this is unlikely, but
       verify rather than assume.
    """
    minimal = parcels[["parcel_id", "geometry"]].copy()

    if buffer_distance > 0:
        if minimal.crs is None or not minimal.crs.is_projected:
            raise ValueError(
                f"CRS is {minimal.crs} -- buffer_distance={buffer_distance} assumes "
                "meters (a projected CRS, e.g. ITM/EPSG:2157)."
            )
        minimal["geometry"] = minimal.geometry.buffer(buffer_distance)

    joined = gpd.sjoin(minimal, minimal, how="inner", predicate="intersects")
    joined = joined.rename(columns={"parcel_id_left": "a", "parcel_id_right": "b"})

    # Dedup: drop self-pairs (a==b, since every geometry intersects
    # itself) and keep only a < b for the reversed-pair duplicates from
    # the self-join.
    edges = joined[joined["a"] < joined["b"]][["a", "b"]].drop_duplicates()

    print(f"Found {len(edges)} adjacency edges for {len(parcels)} parcels "
          f"(avg degree: {2 * len(edges) / len(parcels):.1f}) "
          f"[predicate=intersects, buffer_distance={buffer_distance}m]")
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
