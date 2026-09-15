"""
Masked land-use prediction GNN -- v1 generative model for LandFuture DSS.

TASK: BERT-style masked node prediction over the parcel adjacency graph.
Each parcel's land_cover_class is both an input feature (visible to
neighbors) and the target for itself. A random subset of nodes have
their class replaced with a MASK token each training step; the model
predicts the true class using its own exogenous features plus whatever
signal propagates in from UNMASKED neighbors via message passing.

At inference, this lets you mask an arbitrary region (e.g. a candidate
reallocation zone) and sample plausible land-use fill-ins conditioned
on the surrounding, unmasked landscape -- the "propose realistic
contiguous patterns" piece of the original project brief.

DELIBERATE EXCLUSIONS (leakage prevention):
  carbon_t_c_per_ha_mean and yield_mean are DERIVED FROM land_cover_
  class in this pipeline (tillage always routes to a soil-driven SIS
  number, forestry always gets the flat NFI figure, etc.) -- including
  them as input features would make class prediction near-circular and
  teach the model nothing about real spatial contiguity. Only genuinely
  exogenous features are used: soil association (Associat_S), county,
  region, and (masked) neighbor land-use class.

DEPENDENCIES: torch, torch_geometric. Install per your CUDA version:
  pip install torch torch_geometric
This is a genuinely heavy step -- realistically needs your HPC/GPU
setup at 1.5M nodes / 5.3M edges, not a laptop CPU run.
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

NODES_PATH = "data/graph_nodes.parquet"
EDGES_PATH = "data/graph_edges.parquet"
MODEL_OUT_PATH = "data/masked_gnn_v1.pt"

MASK_RATE = 0.15          # fraction of nodes masked per training step (BERT default)
HIDDEN_DIM = 64
EMBED_DIM = 16
N_EPOCHS = 30
BATCH_SIZE = 2048
NUM_NEIGHBORS = [10, 10]   # 2-hop neighbor sampling fanout, per NeighborLoader batch


# ---------------------------------------------------------------
# 1. LAND-COVER CLASSIFICATION (reused logic from carbon_transition_delta.py)
# ---------------------------------------------------------------

def classify_land_cover(row) -> str:
    if pd.notna(row.get("tillage_class")):
        return "tillage"
    if row.get("is_grassland"):
        return "grassland"
    if row.get("carbon_source") == "nfi_blended_average":
        return "forestry"
    if row.get("carbon_source") == "peatland_literature":
        return "peatland"
    if row.get("is_excluded"):
        return "excluded"
    return "unresolved_crop"


# ---------------------------------------------------------------
# 2. BUILD PYG GRAPH FROM SAVED NODES/EDGES
# ---------------------------------------------------------------

def build_pyg_data():
    nodes = pd.read_parquet(NODES_PATH)
    edges = pd.read_parquet(EDGES_PATH)

    nodes["land_cover_class"] = nodes.apply(classify_land_cover, axis=1)
    classes = sorted(nodes["land_cover_class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    MASK_IDX = len(classes)  # one extra index reserved for the MASK token
    n_classes = len(classes)
    print(f"Land-cover classes: {classes} (MASK token = {MASK_IDX})")

    nodes["class_idx"] = nodes["land_cover_class"].map(class_to_idx)

    # Categorical exogenous features -> integer indices for embedding layers.
    # Fill NA with a dedicated "unknown" category rather than dropping rows.
    for col in ["Associat_S", "COUNTY", "REGION"]:
        nodes[col] = nodes[col].fillna("UNKNOWN").astype(str)
    assoc_categories = sorted(nodes["Associat_S"].unique())
    county_categories = sorted(nodes["COUNTY"].unique())
    assoc_to_idx = {c: i for i, c in enumerate(assoc_categories)}
    county_to_idx = {c: i for i, c in enumerate(county_categories)}
    nodes["assoc_idx"] = nodes["Associat_S"].map(assoc_to_idx)
    nodes["county_idx"] = nodes["COUNTY"].map(county_to_idx)

    # Edge index: undirected -> both directions, as PyG expects.
    edge_index = torch.tensor(
        np.concatenate([
            edges[["a", "b"]].values.T,
            edges[["b", "a"]].values.T,
        ], axis=1),
        dtype=torch.long,
    )

    data = Data(
        edge_index=edge_index,
        class_idx=torch.tensor(nodes["class_idx"].values, dtype=torch.long),
        assoc_idx=torch.tensor(nodes["assoc_idx"].values, dtype=torch.long),
        county_idx=torch.tensor(nodes["county_idx"].values, dtype=torch.long),
        num_nodes=len(nodes),
    )
    # x is a placeholder required by NeighborLoader's sampling machinery;
    # actual features are assembled inside the model from the *_idx
    # tensors above (so masking can be applied per-batch, not baked in).
    data.x = torch.zeros((len(nodes), 1))

    meta = {
        "n_classes": n_classes, "MASK_IDX": MASK_IDX,
        "n_assoc": len(assoc_categories), "n_county": len(county_categories),
        "class_to_idx": class_to_idx,
    }
    return data, meta


# ---------------------------------------------------------------
# 3. MODEL
# ---------------------------------------------------------------

class MaskedLandUseGNN(nn.Module):
    """
    Embeds class (with MASK token), soil association, and county, then
    two SAGEConv layers propagate neighbor information. Output: logits
    over the real land-cover classes (MASK token is never a valid
    prediction target).
    """

    def __init__(self, n_classes, mask_idx, n_assoc, n_county,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.class_embed = nn.Embedding(n_classes + 1, embed_dim)  # +1 for MASK
        self.assoc_embed = nn.Embedding(n_assoc, embed_dim)
        self.county_embed = nn.Embedding(n_county, embed_dim)
        self.mask_idx = mask_idx

        in_dim = embed_dim * 3
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_classes)

    def forward(self, class_idx, assoc_idx, county_idx, edge_index):
        x = torch.cat([
            self.class_embed(class_idx),
            self.assoc_embed(assoc_idx),
            self.county_embed(county_idx),
        ], dim=-1)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x)


# ---------------------------------------------------------------
# 4. TRAINING (mini-batch, neighbor sampling)
# ---------------------------------------------------------------

def train():
    data, meta = build_pyg_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Class-weighted loss: unweighted cross-entropy on this graph
    # collapsed toward predicting the majority class (grassland, 71%
    # of parcels) -- confirmed via masked-accuracy evaluation:
    # grassland 98.0% accuracy vs forestry 5.8%, excluded 5.0%,
    # tillage 28.6%. Since forestry/peatland are the highest-carbon
    # classes in this project, a model that can't predict them is a
    # generative model that can't propose the reallocations that
    # matter most. Inverse-frequency weighting counteracts this by
    # making errors on rare classes cost more during training.
    class_counts = torch.bincount(data.class_idx, minlength=meta["n_classes"])
    class_weights = class_counts.sum() / (meta["n_classes"] * class_counts.float())
    class_weights = class_weights.to(device)
    print("Class weights (inverse frequency):")
    for cls, idx in meta["class_to_idx"].items():
        print(f"  {cls}: count={class_counts[idx].item()}, weight={class_weights[idx].item():.2f}")

    model = MaskedLandUseGNN(
        meta["n_classes"], meta["MASK_IDX"], meta["n_assoc"], meta["n_county"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loader = NeighborLoader(
        data, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE, shuffle=True,
    )

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            batch = batch.to(device)
            true_class = batch.class_idx.clone()

            # Dynamic masking: re-sampled every batch (BERT-style), only
            # on the batch's "seed" nodes (batch.batch_size), not the
            # sampled neighbor context -- masking neighbors too would
            # starve the model of any real signal to propagate from.
            mask = torch.rand(batch.batch_size, device=device) < MASK_RATE
            input_class = batch.class_idx.clone()
            input_class[:batch.batch_size][mask] = meta["MASK_IDX"]

            optimizer.zero_grad()
            logits = model(input_class, batch.assoc_idx, batch.county_idx, batch.edge_index)

            seed_logits = logits[:batch.batch_size][mask]
            seed_targets = true_class[:batch.batch_size][mask]
            if seed_targets.numel() == 0:
                continue

            loss = F.cross_entropy(seed_logits, seed_targets, weight=class_weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{N_EPOCHS}: mean masked-node loss = "
              f"{total_loss / max(n_batches, 1):.4f}  ({epoch_time:.1f}s)")

        # Epoch 1 includes CUDA/cuDNN warmup and lazy-init overhead --
        # not representative. Extrapolate from epoch 2 instead.
        if epoch == 1:
            remaining = N_EPOCHS - 2
            est_remaining_min = epoch_time * remaining / 60
            print(f"[Estimate] Epoch 2 took {epoch_time:.1f}s. At this rate, "
                  f"the remaining {remaining} epochs should take roughly "
                  f"{est_remaining_min:.1f} more minutes "
                  f"(total ≈ {(epoch_time * N_EPOCHS) / 60:.1f} min, "
                  "excluding epoch 1's warmup overhead).")

    torch.save({"model_state": model.state_dict(), "meta": meta}, MODEL_OUT_PATH)
    print(f"Saved model + metadata to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    train()
