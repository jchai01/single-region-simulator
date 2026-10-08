"""
Can data/masked_gnn_v1.pt be trusted to have been trained on the graph files
currently in data/?

Two cases:
  * Checkpoint HAS a graph fingerprint (trained with the current
    train_masked_gnn.py): compared directly against the files on disk.
    This is conclusive.
  * Checkpoint has NO fingerprint (trained before provenance tracking): the
    only evidence available is file timestamps, which can show a checkpoint
    is stale but can never prove it is fresh. The script says so.

Usage (from the repo root):
    python check_gnn_provenance.py [TRAINING_MINUTES]

TRAINING_MINUTES (optional): how long your training run took (train_masked_
gnn.py prints an estimate after epoch 2 and per-epoch times). Training reads
the graph at its START, so the checkpoint must be newer than the edges file
by at least this long to have started after the edges file was written.

Caveat: timestamps are unreliable if files were copied between machines
without preserving mtimes (plain cp, some scp/rsync flags, git checkouts).
"""
import ast
import datetime
import os
import sys

import pyarrow.parquet as pq
import torch

NODES_PATH = "data/graph_nodes.parquet"
EDGES_PATH = "data/graph_edges.parquet"
COMBINED_PATH = "data/parcel_yield_carbon.gpkg"
MODEL_PATH = "data/masked_gnn_v1.pt"
GRAPH_SCRIPT = "build_parcel_graph.py"

try:
    from train_masked_gnn import graph_fingerprint
except Exception:  # patched train script not installed yet, or torch_geometric missing
    graph_fingerprint = None


def sjoin_predicates(src):
    """(line, value) for every real call keyword `predicate="..."`. Parsed with
    ast, so mentions inside docstrings/comments (e.g. the fixed script's
    history of the touches() bug) are NOT counted."""
    found = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (kw.arg == "predicate" and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)):
                    found.append((kw.value.lineno, kw.value.value))
    return sorted(found)


def calls_buffer(src):
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "buffer"
               for n in ast.walk(ast.parse(src)))


def stamp(path):
    t = os.path.getmtime(path)
    return t, datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def main():
    training_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else None
    problems, notes = [], []

    for p in (NODES_PATH, EDGES_PATH, MODEL_PATH):
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p} -- run from the repo root, after building the graph and training.")

    n_nodes = pq.ParquetFile(NODES_PATH).metadata.num_rows
    n_edges = pq.ParquetFile(EDGES_PATH).metadata.num_rows
    avg_degree = 2 * n_edges / n_nodes
    print(f"Graph on disk: {n_nodes:,} nodes, {n_edges:,} edges, avg degree {avg_degree:.2f}")
    if avg_degree < 10:
        problems.append("Average degree is low: this looks like the OLD touches() graph (mean degree ~3 "
                        "when measured on the Offaly subgraph). Rebuild with the corrected "
                        "build_parcel_graph.py first.")
    else:
        print("  -> consistent with the corrected intersects()+buffer graph (16.08M edges / mean degree 20.87 in your NSGA-II logs)")

    if os.path.exists(GRAPH_SCRIPT):
        src = open(GRAPH_SCRIPT).read()
        try:
            preds = sjoin_predicates(src)
            has_buffer = calls_buffer(src)
        except SyntaxError as e:
            preds, has_buffer = [], False
            notes.append(f"Could not parse {GRAPH_SCRIPT} ({e}); skipped the predicate check.")
        shown = ", ".join(f"line {ln}: {v!r}" for ln, v in preds) or "none found"
        print(f"{GRAPH_SCRIPT}: predicate= in actual calls -> {shown}; geometry .buffer() call: {has_buffer}")
        if preds and all(v == "touches" for _, v in preds):
            problems.append(f"{GRAPH_SCRIPT} calls sjoin with predicate='touches' only (the buggy version).")
        elif not preds:
            notes.append(f"Could not find a literal predicate= in {GRAPH_SCRIPT}'s calls; check by hand: "
                         f"grep -n sjoin {GRAPH_SCRIPT}")

    print("\nFile timestamps:")
    t_model, s_model = stamp(MODEL_PATH)
    t_edges, s_edges = stamp(EDGES_PATH)
    t_nodes, s_nodes = stamp(NODES_PATH)
    print(f"  {COMBINED_PATH if os.path.exists(COMBINED_PATH) else '(combined gpkg not found)'}: "
          f"{stamp(COMBINED_PATH)[1] if os.path.exists(COMBINED_PATH) else '-'}")
    print(f"  {NODES_PATH}: {s_nodes}")
    print(f"  {EDGES_PATH}: {s_edges}")
    print(f"  {MODEL_PATH}: {s_model}")
    if os.path.exists(COMBINED_PATH) and stamp(COMBINED_PATH)[0] > max(t_nodes, t_edges):
        problems.append(f"{COMBINED_PATH} is NEWER than the graph files: rebuild the graph "
                        "(build_parcel_graph.py) before training; parcel_id is a positional index.")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    trained_fp = checkpoint.get("meta", {}).get("graph_fingerprint")

    print("\nVerdict:")
    if trained_fp is not None:
        if graph_fingerprint is None:
            notes.append("Checkpoint has a fingerprint but train_masked_gnn.graph_fingerprint "
                         "could not be imported here to compare it.")
        else:
            current = graph_fingerprint(n_nodes, n_edges)
            mismatched = [k for k in ("n_edges", "nodes_sha256", "edges_sha256")
                          if trained_fp[k] != current[k]]
            if mismatched:
                problems.append(f"Checkpoint was trained on a DIFFERENT graph ({trained_fp['n_edges']:,} "
                                f"edges vs {current['n_edges']:,} now; differs in {mismatched}). Retrain.")
            else:
                print("  VERIFIED: the checkpoint's recorded graph fingerprint matches the files on disk.")
    else:
        if t_model < t_edges:
            problems.append("Checkpoint is OLDER than the current edges file, so it was trained on a "
                            "previous graph. Retrain.")
        else:
            gap_min = (t_model - t_edges) / 60
            if training_minutes is not None and gap_min < training_minutes:
                problems.append(f"Checkpoint is only {gap_min:.0f} min newer than the edges file but training "
                                f"took ~{training_minutes:.0f} min, so training started BEFORE the edges file "
                                "was written. Retrain.")
            elif training_minutes is not None:
                notes.append(
                    f"CONSISTENT: training (~{training_minutes:.0f} min) ended {gap_min / 60:.1f} h after the "
                    "edges file was written, so it must have started after it -- i.e. it read the current "
                    "edges. This rests on the file timestamps being genuine (unreliable if files were copied "
                    "between machines without preserving mtimes). No graph fingerprint is stored, so this is "
                    "inference, not proof; retrain once with the current train_masked_gnn.py if you want it provable."
                )
            else:
                notes.append(
                    f"The checkpoint is newer than the edges file by {gap_min / 60:.1f} h. Re-run with your "
                    "training time in minutes (python check_gnn_provenance.py <minutes>): if training took "
                    "less than that gap, it must have started after the edges file was written. Without a "
                    "stored fingerprint this is inference, not proof."
                )

    for p in problems:
        print(f"  PROBLEM: {p}")
    for n in notes:
        print(f"  NOTE: {n}")
    if not problems and not notes:
        pass
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
