"""
Inference / sampling for the trained masked land-use GNN.

Given a target region (a list of parcel_ids to reallocate), extracts
the 2-hop neighborhood subgraph (matching the model's 2-layer receptive
field from training), masks the target parcels, and SAMPLES candidate
land-use assignments from the model's output distribution -- not
argmax, so repeated calls can produce diverse candidate layouts for a
downstream optimizer to evaluate, rather than one deterministic answer.

CAVEAT (see train_masked_gnn.py's docstring): the Associat_S/COUNTY
index mappings are NOT saved in the checkpoint -- they are
deterministically reconstructed by re-running build_pyg_data() on the
same underlying parquet files. This is safe only as long as those
files are unchanged since training. If you retrain after any upstream
data change, re-verify this assumption.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import k_hop_subgraph

from train_masked_gnn import build_pyg_data, MaskedLandUseGNN

MODEL_PATH = "data/masked_gnn_v1.pt"
N_HOPS = 2  # must match the model's number of SAGEConv layers


def load_model_and_graph():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    meta = checkpoint["meta"]

    data, rebuilt_meta = build_pyg_data()
    # Sanity check: the reconstructed graph's class/vocab sizes must
    # match what the checkpoint was trained with, or the embedding
    # layers will silently misinterpret indices.
    for key in ["n_classes", "MASK_IDX", "n_assoc", "n_county"]:
        assert meta[key] == rebuilt_meta[key], (
            f"Mismatch in '{key}': checkpoint has {meta[key]}, "
            f"reconstructed graph has {rebuilt_meta[key]}. The underlying "
            "data has likely changed since training -- do not trust "
            "sampling results until this is resolved (retrain, or "
            "confirm the data files are truly identical)."
        )

    model = MaskedLandUseGNN(
        meta["n_classes"], meta["MASK_IDX"], meta["n_assoc"], meta["n_county"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    idx_to_class = {v: k for k, v in meta["class_to_idx"].items()}
    return model, data, meta, idx_to_class


def sample_reallocation(parcel_ids_to_mask: list[int], model, data, meta, idx_to_class,
                         n_samples: int = 5, temperature: float = 1.0) -> pd.DataFrame:
    """
    parcel_ids_to_mask: the target region's parcel_id values (must match
        the parcel_id assigned in build_parcel_graph.py -- the row
        position in graph_nodes.parquet).
    n_samples: number of independent candidate layouts to draw. Each
        is a full, separate sample from the model's predicted
        distribution -- NOT n_samples forward passes averaged together.
    temperature: >1.0 flattens the distribution (more diverse/random
        samples), <1.0 sharpens it toward the model's most confident
        guess. 1.0 = use the model's raw predicted probabilities.

    Returns a DataFrame: one row per (sample_id, parcel_id), with the
    sampled class label and its predicted probability under the model.
    """
    seed_nodes = torch.tensor(parcel_ids_to_mask, dtype=torch.long)

    # Extract the k-hop subgraph around the masked region -- this is
    # what makes inference cheap even though the full graph has 1.5M
    # nodes: we only need the masked parcels' actual receptive field,
    # not the whole country, matching the model's 2-layer architecture.
    sub_nodes, sub_edge_index, mapped_seed_idx, _ = k_hop_subgraph(
        seed_nodes, N_HOPS, data.edge_index, relabel_nodes=True,
    )

    sub_class_idx = data.class_idx[sub_nodes].clone()
    sub_assoc_idx = data.assoc_idx[sub_nodes]
    sub_county_idx = data.county_idx[sub_nodes]

    # Mask exactly the target parcels within the subgraph. Neighbors
    # stay unmasked -- they're the real context the model conditions on.
    input_class = sub_class_idx.clone()
    input_class[mapped_seed_idx] = meta["MASK_IDX"]

    with torch.no_grad():
        logits = model(input_class, sub_assoc_idx, sub_county_idx, sub_edge_index)
        seed_logits = logits[mapped_seed_idx]
        probs = F.softmax(seed_logits / temperature, dim=-1)

    rows = []
    for sample_id in range(n_samples):
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled_prob = probs.gather(1, sampled_idx.unsqueeze(1)).squeeze(1)
        for i, parcel_id in enumerate(parcel_ids_to_mask):
            rows.append({
                "sample_id": sample_id,
                "parcel_id": parcel_id,
                "sampled_class": idx_to_class[sampled_idx[i].item()],
                "sampled_probability": sampled_prob[i].item(),
            })

    return pd.DataFrame(rows)


def evaluate_masked_accuracy(model, data, meta, idx_to_class, n_eval: int = 2000,
                              seed: int = 42, batch_size: int = 256) -> pd.DataFrame:
    """
    Proper validation: randomly mask n_eval individual nodes ACROSS THE
    WHOLE GRAPH (not a cherry-picked or arbitrary contiguous range),
    predict each from its real neighbors, and compare to its TRUE
    original class. Reports overall and per-class accuracy -- this is
    the real test of whether the model learned spatial context, not
    just class frequencies (a high overall accuracy driven entirely by
    always guessing the majority "grassland" class would show up
    clearly as near-zero accuracy on the minority classes below).

    Uses NeighborLoader for proper mini-batch GPU evaluation -- same
    approach as training, rather than one-node-at-a-time k_hop_subgraph
    calls in a Python loop (which is correct but forfeits both GPU
    acceleration and batching speed, since GPUs gain nothing from many
    small sequential calls).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    rng = np.random.default_rng(seed)
    eval_node_ids = torch.tensor(
        rng.choice(data.num_nodes, size=n_eval, replace=False), dtype=torch.long,
    )

    loader = NeighborLoader(
        data, num_neighbors=[10, 10], batch_size=batch_size,
        input_nodes=eval_node_ids, shuffle=False,
    )

    correct_by_class = {}
    total_by_class = {}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            true_class = batch.class_idx[:batch.batch_size].clone()

            input_class = batch.class_idx.clone()
            input_class[:batch.batch_size] = meta["MASK_IDX"]

            logits = model(input_class, batch.assoc_idx, batch.county_idx, batch.edge_index)
            pred_idx = logits[:batch.batch_size].argmax(dim=-1)

            for t, p in zip(true_class.tolist(), pred_idx.tolist()):
                true_cls = idx_to_class[t]
                total_by_class[true_cls] = total_by_class.get(true_cls, 0) + 1
                if p == t:
                    correct_by_class[true_cls] = correct_by_class.get(true_cls, 0) + 1

    rows = []
    for cls, total in total_by_class.items():
        correct = correct_by_class.get(cls, 0)
        rows.append({"class": cls, "n": total, "accuracy": correct / total})
    result = pd.DataFrame(rows).sort_values("n", ascending=False)

    overall_acc = sum(correct_by_class.values()) / n_eval
    print(f"\nOverall masked-node accuracy on {n_eval} random parcels: {overall_acc:.1%}")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    model, data, meta, idx_to_class = load_model_and_graph()

    # Example: mask an arbitrary small set of parcels and sample 5
    # candidate reallocations. Replace with a real region of interest
    # (e.g. all parcel_ids within a chosen county/soil association) --
    # this is just a smoke test to confirm the pipeline runs end to end.
    example_parcel_ids = list(range(100, 110))
    results = sample_reallocation(example_parcel_ids, model, data, meta, idx_to_class,
                                   n_samples=5)
    print(results.to_string(index=False))

    print("\n--- Class distribution across samples (per parcel) ---")
    for pid in example_parcel_ids:
        subset = results[results["parcel_id"] == pid]
        print(f"parcel_id {pid}: {subset['sampled_class'].value_counts().to_dict()}")

    print("\n--- Per-class masked-accuracy evaluation (the real test) ---")
    evaluate_masked_accuracy(model, data, meta, idx_to_class, n_eval=2000)
