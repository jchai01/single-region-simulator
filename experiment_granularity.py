"""
Experiment 2b -- when does greedy stop being near-optimal? (decision-unit size)

Greedy is near-optimal here because the decision units (single parcels) are tiny
relative to the disruption budget: the only thing a knapsack greedy can lose is the
lumpiness of the last item. If that is the whole story, greedy's gap must GROW as the
units get coarser -- which is exactly the regime where a search method has something
to find. This experiment measures that.

Decision units are BLOCKS: neighbouring parcels grown together over the real adjacency
graph (a proxy for a farm holding), each block taking ONE class for all its parcels. A
block's gain is the sum of its parcels' gains for that class; its weight is the area of
the parcels whose class actually changes. Block size 1 is the ordinary per-parcel
problem (and must reproduce the headline numbers).

Two greedy rules are compared with the exact optimum:
  stop  -- density-greedy that stops at the first block that would overshoot the budget
           (what baseline_comparison.greedy_baseline does)
  fill  -- the same, but skips that block and keeps scanning (a standard stronger heuristic)

The block-size sweep is repeated over random partitions (different seeds give different
blocks) so the spread, not just the mean, is visible.

Usage (repo root):  python experiment_granularity.py [--partitions 10] [--sizes 1 2 3 5 8 12 20 35 60]
"""
import argparse
import os

import numpy as np
import pandas as pd

from mckp_tools import (block_instance, class_gain_tables, edges_to_index_pairs, make_blocks,
                        solve_exact, solve_greedy)

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SIZES = [1, 2, 3, 5, 8, 12, 20, 35, 60]


def run(solver, edge_pairs, sizes=SIZES, partitions=10, budgets=BUDGETS, seed=1):
    G, W, _ = class_gain_tables(solver)
    rng = np.random.default_rng(seed)
    rows = []
    for m in sizes:
        for r in range(1 if m == 1 else partitions):
            blocks = [np.array([i]) for i in range(solver.n)] if m == 1 else make_blocks(solver.n, edge_pairs, m, rng)
            inst, _, _ = block_instance(G, W, blocks, solver.total_area)
            largest = max(float(solver.area[b].sum()) for b in blocks)
            for b in budgets:
                _, opt = solve_exact(inst, b)
                _, g_stop = solve_greedy(inst, b)
                _, g_fill = solve_greedy(inst, b, fill=True)
                rows.append({"block_size": m, "rep": r, "n_blocks": len(blocks), "budget": b, "opt": opt,
                             "greedy_stop": g_stop, "greedy_fill": g_fill,
                             "gap_stop_pct": 100 * (1 - g_stop / opt) if opt > 0 else np.nan,
                             "gap_fill_pct": 100 * (1 - g_fill / opt) if opt > 0 else np.nan,
                             "largest_block_pct_of_budget": 100 * largest / (b * solver.total_area),
                             # when even the smallest useful move does not fit the budget, optimum = greedy = 0 and the
                             # gap is undefined (NaN) -- reported via fits_pct instead of being silently averaged away
                             "fits": opt > 0})
    return pd.DataFrame(rows)


def plot(df, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("(matplotlib unavailable; skipping figure)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colours = {0.1: "#534AB7", 0.3: "#993C1D", 0.5: "#2E5E3E"}
    for b, c in colours.items():
        sub = df[df["budget"] == b]
        for col, ls, lab in (("gap_stop_pct", "-", "stop at first overshoot"), ("gap_fill_pct", "--", "skip and keep filling")):
            g = sub.groupby("block_size")[col]
            ax.plot(g.mean().index, g.mean().values, ls=ls, marker="o", color=c, label=f"budget {b:g}, {lab}")
        g = sub.groupby("block_size")["gap_stop_pct"]
        ax.fill_between(g.max().index, g.min().values, g.max().values, color=c, alpha=0.10)
    ax.set_xscale("log")
    ax.set_xlabel("decision-unit size (parcels per block)"); ax.set_ylabel("greedy's gap to the exact optimum (%)")
    ax.set_title("Greedy's gap to the exact optimum vs decision-unit size\n(lines = mean over random block partitions; shading = range, 'stop' rule)")
    ax.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Wrote {out_path}")


def summarise(df):
    keep = df[df["budget"].isin([0.1, 0.3, 0.5])]
    p = keep.pivot_table(index="block_size", columns="budget", values=["gap_stop_pct", "gap_fill_pct"], aggfunc="mean")
    p.columns = [f"{'stop' if 'stop' in a else 'fill'} gap % @{b:g}" for a, b in p.columns]
    extra = df.groupby("block_size").agg(blocks=("n_blocks", "mean"),
                                         fits_pct=("fits", lambda x: 100 * x.mean()),
                                         worst_stop_gap_pct=("gap_stop_pct", "max"),
                                         worst_fill_gap_pct=("gap_fill_pct", "max"))
    item = df[df["budget"] == 0.1].groupby("block_size")["largest_block_pct_of_budget"].mean().rename("largest block, % of the 0.1 budget")
    return p.join(extra).join(item)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=10)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    import solve_exact_front as sef
    from nsga2_scenario_search import evaluate_batch

    problem = sef.build_problem()
    solver = sef.ExactSolver(problem, evaluate_batch, True)
    edges = pd.read_parquet("data/graph_edges.parquet")
    pairs = edges_to_index_pairs(edges, problem.parcel_ids)
    deg = np.bincount(np.array(pairs).ravel(), minlength=solver.n) if pairs else np.zeros(solver.n)
    print(f"{solver.n} candidate parcels; {len(pairs):,} adjacencies among them (mean degree {deg.mean():.1f}, "
          f"{int((deg == 0).sum())} isolated)", flush=True)

    df = run(solver, pairs, a.sizes, a.partitions, seed=a.seed)

    # consistency: block size 1 must reproduce the verified per-parcel optimum
    base = df[df["block_size"] == 1].set_index("budget")["opt"]
    worst = max(abs(base[b] - solver.solve(b)["carbon_delta_t"]) / solver.solve(b)["carbon_delta_t"] for b in base.index)
    print(f"Check: block size 1 reproduces ExactSolver's optimum at every budget (max relative difference {worst:.1e}).")

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/granularity_gap.csv", index=False)
    pd.set_option("display.width", 240); pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("\n=== Greedy's gap to the exact optimum (%), by decision-unit size ===")
    print("(gaps are averaged over the cases where at least one block fits the budget; fits_pct is the share of such cases)")
    print(summarise(df).to_string())
    print("\nWrote data/granularity_gap.csv")
    if not a.no_plot:
        plot(df, "figures/granularity_gap.png")


if __name__ == "__main__":
    main()
