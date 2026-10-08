"""
Figures for the manuscript, regenerated from the repository's own result files.

    python make_paper_figures.py [--data-dir data] [--out-dir figures]

Reads (all written by the pipeline): exact_front.csv, gain_uncertainty_generic.csv,
gain_uncertainty_data.csv, granularity_gap.csv.

Two sets of numbers in Figure 2 are NOT in any CSV and are transcribed from printed output of the
final run, so update them if you rerun (they are marked TRANSCRIBED below):
  * the exact optima and the greedy / NSGA-II values per budget (printed by solve_exact_front.py
    and compare_methods_at_budget.py; the optima are also in exact_comparison.csv),
  * the greedy min/max over tie orders (printed by check_greedy_agreement.py) and the per-seed
    NSGA-II values (printed by compare_methods_at_budget.py).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data")
ap.add_argument("--out-dir", default="figures")
_a = ap.parse_args()
DATA = _a.data_dir.rstrip("/") + "/"; OUT = _a.out_dir
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

def figure1():
    C1 = {"beige": ("#F1EEE6", "#8A867B"), "mint": ("#E1F3EC", "#2E8B6B"), "lav": ("#EEECFB", "#6F63D1"), "peach": ("#F9ECE6", "#C26A4A")}
    fig, ax = plt.subplots(figsize=(9.0, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.axis("off")
    def box(x, y, w, h, title, sub="", kind="beige", fs=8.8, dashed=False):
        fc, ec = C1[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.3", fc=fc, ec=ec, lw=1.0, ls="--" if dashed else "-"))
        ax.text(x + w/2, y + h*(0.64 if sub else 0.5), title, ha="center", va="center", fontsize=fs, fontweight="bold", color="#222")
        if sub: ax.text(x + w/2, y + h*0.27, sub, ha="center", va="center", fontsize=fs-1.9, color="#555")
    def arrow(x1, y1, x2, y2, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=9, lw=1.1, color="#444", ls="--" if dashed else "-"))
    # row 1
    box(1, 58, 26, 11, "LPIS + SIS data", "1.54M parcels nationally", "beige")
    box(37, 58, 26, 11, "Carbon model", "SOLUM-hybrid, 25-year", "mint")
    box(73, 58, 26, 11, "Candidate set", "611 parcels in Offaly", "mint")
    arrow(27.5, 63.5, 36.5, 63.5); arrow(63.5, 63.5, 72.5, 63.5)
    # row 2
    box(1, 40, 26, 12, "Graph + masked GNN", "corrected adjacency", "lav")
    ax.add_patch(FancyBboxPatch((36, 39.2), 28, 14, boxstyle="round,pad=0.25,rounding_size=1.3", fc="white", ec="#AAA", lw=1, ls="--"))
    ax.text(50, 50.8, "Independent validation", ha="center", fontsize=7.6, color="#555")
    box(38, 41, 12.2, 7, "Spatial", "", "beige", fs=7.8); box(50.8, 41, 12.2, 7, "Temporal", "", "beige", fs=7.8)
    arrow(14, 57.6, 14, 52.6); arrow(50, 57.6, 50, 53.9, dashed=True)
    # row 3: methods
    ax.add_patch(FancyBboxPatch((1, 17.5), 98, 17.5, boxstyle="round,pad=0.25,rounding_size=1.3", fc="white", ec="#BBB", lw=1))
    ax.text(50, 32.7, "Methods compared at matched disruption budgets, all scored by one rule (policy_year_stock)", ha="center", fontsize=7.8, color="#555")
    for i, (t, k) in enumerate([("NSGA-II\n(GNN-seeded)", "lav"), ("Random", "beige"), ("Greedy\n(stop)", "peach"), ("Greedy\n(fill)", "peach"), ("Exact optimum\n(MILP)", "mint")]):
        x = 2.4 + i * 19.4
        box(x, 19.2, 17.2, 10.2, "", "", k); ax.text(x + 8.6, 24.3, t, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#222")
    arrow(14, 39.4, 14, 35.4)                 # GNN seeds NSGA-II (leftmost method)
    arrow(86, 57.6, 86, 35.4)                 # candidate set feeds every method
    # row 4
    box(14, 2, 34, 10, "Budget-matched comparison", "% of the exact optimum", "peach", fs=8.6)
    box(56, 2, 34, 10, "Robustness experiments", "gain uncertainty; decision-unit size", "peach", fs=8.6)
    arrow(31, 17.1, 31, 12.6); arrow(48.5, 7, 55.6, 7)
    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figures_2_to_4():
    COL = {"exact": "#222222", "fill": "#2E8B6B", "stop": "#C26A4A", "nsga": "#534AB7", "rand": "#888780"}

    # ======================= Figure 2: benchmark =======================
    b = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    opt = np.array([13996.90, 25851.81, 30030.27, 31979.98, 33764.40, 35548.04])  # TRANSCRIBED from printed output of the final run
    stop = np.array([13770.33, 25713.93, 30010.02, 31943.41, 33706.93, 35536.51]); fill = np.array([13994.02, 25847.20, 30031.75, 31980.43, 33764.46, 35547.80])  # TRANSCRIBED from printed output of the final run
    stop_lo = np.array([12646.65, 25236.16, 29896.72, 31807.90, 33587.54, 35402.39]); stop_hi = np.array([13995.91, 25851.39, 30032.02, 31980.65, 33764.15, 35547.18])  # TRANSCRIBED from printed output of the final run
    fill_lo = np.array([13870.01, 25815.99, 30018.34, 31977.91, 33753.74, 35526.64]); fill_hi = np.array([13997.24, 25852.55, 30032.25, 31980.86, 33764.46, 35548.04])  # TRANSCRIBED from printed output of the final run
    seeds = {0.1: [13768.3, 13832.2, 13914.1, 13877.3], 0.2: [25398.7, 25336.8, 25343.8, 25298.3], 0.3: [29816.7, 29798.3, 29723.2, 29850.7],  # TRANSCRIBED from printed output of the final run
             0.4: [31792.4, 31778.6, 31689.5, 31802.0], 0.5: [33608.0, 33551.5, 33470.3, 33560.0]}
    seeds06 = [(0.547, 34451.8), (0.550, 34464.3), (0.551, 34321.4), (0.564, 34637.8)]       # (disruption reached, carbon): budget 0.6 not reached  # TRANSCRIBED from printed output of the final run
    front = pd.read_csv(DATA + "exact_front.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1, 1.3]})
    ax[0].plot(front.budget, front.as_searched / 1000, color=COL["exact"], lw=1.6, label="Exact optimum")
    ax[0].scatter(b, fill / 1000, s=34, color=COL["fill"], zorder=3, label="Greedy (fill)")
    ax[0].scatter(b, stop / 1000, s=30, marker="s", facecolor="none", edgecolor=COL["stop"], zorder=3, label="Greedy (stop)")
    nb = [max(seeds[x]) for x in b[:5]]
    ax[0].scatter(b[:5], np.array(nb) / 1000, s=36, marker="^", color=COL["nsga"], zorder=3, label="NSGA-II (best of 4 seeds)")
    ax[0].scatter([0.564], [34.6378], s=36, marker="^", facecolor="none", edgecolor=COL["nsga"], zorder=3)
    ax[0].annotate("budget 0.6 not reached:\nfront ends at 0.564", (0.564, 34.64), (0.30, 19.0), fontsize=6.0, arrowprops=dict(arrowstyle="-", color="#999", lw=0.7), color="#555")
    ax[0].set_xlabel("Disruption budget (fraction of region area changed)"); ax[0].set_ylabel("Carbon gain at year 25 (kt C)")
    ax[0].set_title("(a) all methods sit on the exact front", fontsize=8.2, loc="left"); ax[0].legend(frameon=False, fontsize=6.0, loc="lower right"); ax[0].set_xlim(0, 0.72)
    off = {"fill": -0.011, "stop": 0.0, "nsga": 0.011}
    pct = lambda v, i: 100 * v / opt[i]
    for i, bb in enumerate(b):
        ax[1].plot([bb + off["fill"]] * 2, [pct(fill_lo[i], i), pct(fill_hi[i], i)], color=COL["fill"], lw=2.2, alpha=0.55, solid_capstyle="round")
        ax[1].plot([bb + off["stop"]] * 2, [pct(stop_lo[i], i), pct(stop_hi[i], i)], color=COL["stop"], lw=2.2, alpha=0.45, solid_capstyle="round")
    ax[1].scatter(b + off["fill"], 100 * fill / opt, s=30, color=COL["fill"], zorder=3, label="Greedy (fill); bar = all tie orders")
    ax[1].scatter(b + off["stop"], 100 * stop / opt, s=28, marker="s", facecolor="white", edgecolor=COL["stop"], zorder=3, label="Greedy (stop); bar = all tie orders")
    for i, bb in enumerate(b[:5]):
        v = np.array(seeds[bb]); ax[1].scatter([bb + off["nsga"]] * 4, 100 * v / opt[i], s=11, color=COL["nsga"], alpha=0.45, zorder=2)
    ax[1].scatter(b[:5] + off["nsga"], [100 * np.mean(seeds[x]) / opt[i] for i, x in enumerate(b[:5])], s=34, marker="^", color=COL["nsga"], zorder=3, label="NSGA-II: seeds (dots), mean (triangle)")
    ex564 = 34905.9  # TRANSCRIBED from printed output of the final run
    ax[1].scatter([0.564], [100 * 34637.8 / ex564], s=34, marker="^", facecolor="none", edgecolor=COL["nsga"], zorder=3)
    ax[1].axhline(100, color="#999", lw=0.8)
    ax[1].set_ylim(89.5, 100.7); ax[1].set_xlim(0.05, 0.66)
    ax[1].set_xlabel("Disruption budget"); ax[1].set_ylabel("% of the exact optimum")
    ax[1].set_title("(b) gap to the optimum, with sensitivity to tie order", fontsize=8.2, loc="left")
    ax[1].text(0.575, 98.55, "open triangle: best seed at its own\ndisruption (0.564), vs the exact\noptimum there (99.2%)", fontsize=5.7, color="#555")
    ax[1].legend(frameon=False, fontsize=5.8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_benchmark.png"), dpi=220, facecolor="white"); plt.close(fig)

    # ======================= Figure 3: uncertainty =======================
    def summarise(df):
        df = df.copy(); df["A"] = 100 * df.greedy_informed / df.opt_true; df["ex"] = 100 * df.exact_plan_realised / df.opt_true; df["gr"] = 100 * df.greedy_plan_realised / df.opt_true
        df["diff"] = df.ex - df.gr
        return df
    gen = summarise(pd.read_csv(DATA + "gain_uncertainty_generic.csv")); dat = summarise(pd.read_csv(DATA + "gain_uncertainty_data.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4))
    cols = {0.1: "#9DB8F0", 0.2: "#5B7FD6", 0.3: "#2A47A8"}
    for s, c in cols.items():
        g = gen[gen.sigma == s].groupby("budget").A; ax[0].plot(g.mean().index, g.mean().values, marker="o", ms=3.5, color=c, lw=1.3, label=f"generic σ = {s:g}")
        ax[0].fill_between(g.mean().index, g.min().values, g.mean().values, color=c, alpha=0.12, lw=0)
    g = dat.groupby("budget").A; ax[0].plot(g.mean().index, g.mean().values, marker="s", ms=4, color=COL["fill"], lw=1.6, label="data-driven σ (median 6.1%)")
    ax[0].fill_between(g.mean().index, g.min().values, g.mean().values, color=COL["fill"], alpha=0.18, lw=0)
    ax[0].axhline(100, color="#999", lw=0.8); ax[0].set_ylim(99.4, 100.06)
    ax[0].set_xlabel("Disruption budget"); ax[0].set_ylabel("Greedy (fill), % of optimum\non the perturbed gains")
    ax[0].set_title("(a) perturbed gains: greedy (fill)\n(mean of 100 draws; shading = worst draw)", fontsize=8, loc="left"); ax[0].legend(frameon=False, fontsize=6.3, loc="lower right")
    t = dat.groupby("budget").agg(regret=("ex", lambda s: 100 - s.mean()), diff=("diff", "mean"))
    x = np.arange(6); w = 0.36
    ax[1].bar(x - w/2, t.regret, w, color="#B9B7AE", label="loss from planning on estimates")
    ax[1].bar(x + w/2, t["diff"].clip(lower=0), w, color=COL["stop"], label="exact plan minus greedy (fill) plan")
    for i, (r, d) in enumerate(zip(t.regret, t["diff"])):
        ax[1].text(i - w/2, r + 0.25, f"{r:.2f}", ha="center", fontsize=6.0, color="#555"); ax[1].text(i + w/2, max(d, 0) + 0.25, f"{abs(round(max(d, 0), 2)):.2f}", ha="center", fontsize=6.0, color=COL["stop"])
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{v:.1f}" for v in t.index]); ax[1].set_xlabel("Disruption budget"); ax[1].set_ylabel("percentage points of the optimum")
    ax[1].set_title("(b) data-driven uncertainty:\ncost of uncertainty vs optimiser gain", fontsize=8, loc="left"); ax[1].legend(frameon=False, fontsize=6.2, loc="upper right"); ax[1].set_ylim(0, 13.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_uncertainty.png"), dpi=220, facecolor="white"); plt.close(fig)

    # ======================= Figure 4: granularity =======================
    g = pd.read_csv(DATA + "granularity_gap.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    bc = {0.1: "#534AB7", 0.3: "#C26A4A", 0.5: "#2E8B6B"}
    for k, (col, ttl) in enumerate((("gap_fill_pct", "(a) greedy with fill"), ("gap_stop_pct", "(b) greedy that stops at the first overshoot"))):
        for bud, c in bc.items():
            s = g[g.budget == bud].groupby("block_size")[col]
            ax[k].plot(s.mean().index, s.mean().values, marker="o", ms=3.8, color=c, lw=1.5, label=f"budget {bud:g}: mean")
            ax[k].plot(s.max().index, s.max().values, color=c, lw=0.9, ls=":", label=f"budget {bud:g}: worst partition")
        ax[k].set_xscale("log"); ax[k].set_xticks([1, 2, 3, 5, 8, 12, 20, 35, 60]); ax[k].set_xticklabels(["1", "2", "3", "5", "8", "12", "20", "35", "60"])
        ax[k].set_xlabel("Decision-unit size (parcels per block)"); ax[k].set_title(ttl, fontsize=8.2, loc="left"); ax[k].set_ylim(-1, 45)
    ax[0].set_ylabel("Gap to the exact optimum (%)"); ax[1].legend(frameon=False, fontsize=5.7, loc="upper left", ncol=1)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_granularity.png"), dpi=220, facecolor="white"); plt.close(fig)



if __name__ == "__main__":
    figure1()
    figures_2_to_4()
    print(f"figures written to {OUT}/")
