"""
Empirically tests whether GNN-fallback rate correlates with interiority
(fraction of a parcel's own 1-hop neighbors that are ALSO inside the
masked candidate region) -- the mechanism suggested by sample_
reallocation.py's actual masking logic: k_hop_subgraph() pulls the
2-hop neighborhood of the WHOLE region at once, and every parcel in
parcel_ids_to_mask is masked SIMULTANEOUSLY. A parcel whose entire
2-hop neighborhood is ALSO in the masked set has no real neighbor
LAND-USE signal feeding its prediction (though assoc_idx/county_idx
remain real, since only class_idx is masked).
"""

import pandas as pd
from sample_reallocation import load_model_and_graph, sample_reallocation

REALISTIC_TARGET_CLASSES = ["tillage", "grassland", "forestry"]
EDGES_PATH = "data/graph_edges.parquet"


def compute_interiority(parcel_ids, edges_path=EDGES_PATH):
    edges = pd.read_parquet(edges_path)
    region_set = set(parcel_ids)
    region_edges = edges[edges["a"].isin(region_set) | edges["b"].isin(region_set)]

    neighbor_lists = {}
    for a, b in zip(region_edges["a"], region_edges["b"]):
        if a in region_set:
            neighbor_lists.setdefault(a, []).append(b)
        if b in region_set:
            neighbor_lists.setdefault(b, []).append(a)

    interiority = {}
    for pid, neighbors in neighbor_lists.items():
        if not neighbors:
            continue
        interiority[pid] = sum(1 for n in neighbors if n in region_set) / len(neighbors)
    return interiority


def is_peat(pid, combined):
    row = combined[combined["parcel_id"] == pid]
    if len(row) == 0:
        return False
    return row.iloc[0]["land_cover_class"] == "peatland"


if __name__ == "__main__":
    with open("data/offaly_subregion_parcel_ids.txt") as f:
        parcel_ids = [int(line.strip()) for line in f if line.strip()]

    print("Loading model and drawing real GNN proposals for this region...")
    model, data, meta, idx_to_class = load_model_and_graph()
    samples = sample_reallocation(parcel_ids, model, data, meta, idx_to_class,
                                   n_samples=10, temperature=1.0)

    import geopandas as gpd
    from carbon_transition_delta import classify_land_cover
    combined = gpd.read_file("data/parcel_yield_carbon.gpkg").reset_index(drop=True)
    combined["parcel_id"] = combined.index
    combined["land_cover_class"] = combined.apply(classify_land_cover, axis=1)
    combined = combined[["parcel_id", "land_cover_class"]]

    def valid_labels_for(pid):
        return REALISTIC_TARGET_CLASSES + (["peatland"] if is_peat(pid, combined) else [])

    samples["fell_back"] = samples.apply(
        lambda r: r["sampled_class"] not in valid_labels_for(r["parcel_id"]), axis=1
    )

    # Per-parcel fallback rate ACROSS ALL 10 samples (not per-sample --
    # since probs is computed once per parcel and reused for every draw,
    # a parcel's fallback tendency is a property of its OWN predicted
    # distribution, not something that varies much sample-to-sample)
    per_parcel_fallback = samples.groupby("parcel_id")["fell_back"].mean()

    interiority = pd.Series(compute_interiority(parcel_ids))

    merged = pd.DataFrame({
        "interiority": interiority,
        "fallback_rate": per_parcel_fallback,
    }).dropna()

    print(f"\n{len(merged)} parcels with both interiority and fallback data")
    merged["interiority_bucket"] = pd.cut(
        merged["interiority"], [0, 0.3, 0.6, 0.8, 1.01],
        labels=["low (<0.3)", "medium (0.3-0.6)", "high (0.6-0.8)", "very high (>0.8)"]
    )
    print("\nMean fallback rate by interiority bucket:")
    print(merged.groupby("interiority_bucket")["fallback_rate"].agg(["mean", "count"]).to_string())

    correlation = merged["interiority"].corr(merged["fallback_rate"])
    print(f"\nCorrelation (interiority vs fallback_rate): {correlation:.3f}")
    print("(strong positive correlation confirms the context-starvation "
          "hypothesis; weak/no correlation means something else is driving it)")
