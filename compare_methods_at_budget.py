"""
Budget-matched comparison of NSGA-II vs greedy vs random, from the exported
per-parcel scenario files (data/scenario_details_seed*.parquet and
data/scenario_details_baselines.parquet).

THE COMPARISON THIS REPLACES WAS BIASED. Reading NSGA-II's value for a
disruption level off the PRINTED front means taking the best carbon among
points whose disruption ROUNDS to that level -- e.g. every point from 0.05
to 0.149 counts as "0.1". Carbon rises with disruption, so the best point in
that bucket is the one with the MOST disruption (up to ~50% over budget at
the 0.1 level), while greedy is evaluated at exactly 0.1. That hands NSGA-II
free disruption and inflates its advantage.

The fair question is: what is the best carbon each method achieves WITHIN a
disruption budget d? For NSGA-II that is the best front point with
disruption <= d; for greedy it is its scenario at budget d (its own actual
disruption is <= d by construction). The exports store the exact, unrounded
disruption_fraction, so this can be computed exactly.

For contrast the script also prints the old rounded-bucket figure, so the size
of the bias is visible rather than asserted.

Usage (from the repo root):  python compare_methods_at_budget.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
EPS = 1e-9
REACH_TOL = 0.01   # a seed "reaches" budget d if its front extends to >= d - REACH_TOL


def check_same_candidate_set(files):
    """Every export must cover exactly the same parcels. Exports built on different candidate sets
    (e.g. some stages run with the tillage-like reclassification and some without) are different
    problems, and comparing them silently would be wrong."""
    sets = {f: frozenset(pd.read_parquet(f, columns=["parcel_id"])["parcel_id"].unique()) for f in files}
    ref_file = files[0]
    bad = [f for f in files if sets[f] != sets[ref_file]]
    if bad:
        lines = [f"  {f}: {len(sets[f]):,} parcels" for f in files]
        raise SystemExit(
            "These exports were built on DIFFERENT candidate sets and must not be compared:\n" + "\n".join(lines) +
            "\nRerun every stage (baseline_comparison.py and all NSGA-II seeds) with the same setting of "
            "generate_scenarios.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED (each run prints which one it used).")


def load_scenarios(nsga_glob="data/scenario_details_seed*.parquet",
                   baselines_path="data/scenario_details_baselines.parquet"):
    """One row per (method, scenario_id) -- the exports repeat the scenario's
    two objectives on every parcel row."""
    cols = ["method", "scenario_id", "disruption_fraction", "carbon_delta_vs_baseline_t"]
    files = sorted(glob.glob(nsga_glob))
    if os.path.exists(baselines_path):
        files.append(baselines_path)
    else:
        print(f"(no {baselines_path}; run baseline_comparison.py to include greedy/random)")
    if not files:
        raise SystemExit("No scenario export files found in data/.")
    check_same_candidate_set(files)
    parts = [pd.read_parquet(f, columns=cols) for f in files]
    sc = pd.concat(parts, ignore_index=True).drop_duplicates(["method", "scenario_id"])
    return sc


def compare(sc):
    nsga = sc[sc["method"] == "nsga2"].copy()
    nsga["seed"] = nsga["scenario_id"].str.extract(r"seed(\d+)_")[0].astype(int)
    seeds = sorted(nsga["seed"].unique())
    greedy = sc[sc["method"] == "greedy"].set_index(sc[sc["method"] == "greedy"]["scenario_id"].astype(float))
    gfill = sc[sc["method"] == "greedy_fill"]
    gfill = gfill.set_index(gfill["scenario_id"].astype(float))
    random_ = sc[sc["method"] == "random"].copy()
    random_["level"] = random_["scenario_id"].str.extract(r"^([0-9.]+)_seed")[0].astype(float)

    seed_max_disruption = nsga.groupby("seed")["disruption_fraction"].max()
    rows, per_seed = [], []
    for d in BUDGETS:
        best = {}
        for s in seeds:
            sub = nsga[(nsga["seed"] == s) & (nsga["disruption_fraction"] <= d + EPS)]
            if len(sub):
                top = sub.loc[sub["carbon_delta_vs_baseline_t"].idxmax()]
                best[s] = (top["carbon_delta_vs_baseline_t"], top["disruption_fraction"])
        reached = [s for s in seeds if seed_max_disruption[s] >= d - REACH_TOL]
        vals_all = np.array([v[0] for v in best.values()])
        vals_reached = np.array([best[s][0] for s in reached if s in best])
        top_seed = max(best, key=lambda s: best[s][0])

        # the OLD, biased reading: best carbon among points whose disruption ROUNDS to d
        bucket = nsga[nsga["disruption_fraction"].round(1) == d]
        old_bucket = bucket["carbon_delta_vs_baseline_t"].max() if len(bucket) else np.nan

        g = greedy.loc[d] if d in greedy.index else None
        g_carbon = g["carbon_delta_vs_baseline_t"] if g is not None else np.nan
        g_disr = g["disruption_fraction"] if g is not None else np.nan
        gf = gfill.loc[d] if d in gfill.index else None
        gf_carbon = gf["carbon_delta_vs_baseline_t"] if gf is not None else np.nan
        gf_disr = gf["disruption_fraction"] if gf is not None else np.nan
        r = random_[random_["level"] == d]["carbon_delta_vs_baseline_t"]

        rows.append({
            "budget": d,
            "greedy_carbon": g_carbon, "greedy_disruption_used": g_disr,
            "greedy_fill_carbon": gf_carbon, "greedy_fill_disruption_used": gf_disr,
            "nsga2_best_pct_vs_greedy_fill": 100 * (best[top_seed][0] / gf_carbon - 1) if gf is not None else np.nan,
            "nsga2_mean_pct_vs_greedy_fill": 100 * (vals_reached.mean() / gf_carbon - 1) if len(vals_reached) and gf is not None else np.nan,
            "nsga2_best_seed_carbon": best[top_seed][0], "nsga2_best_disruption_used": best[top_seed][1],
            "nsga2_best_pct_vs_greedy": 100 * (best[top_seed][0] / g_carbon - 1) if g is not None else np.nan,
            "nsga2_mean_reached": vals_reached.mean() if len(vals_reached) else np.nan,
            "nsga2_sd_reached": vals_reached.std(ddof=1) if len(vals_reached) > 1 else np.nan,
            "nsga2_mean_pct_vs_greedy": 100 * (vals_reached.mean() / g_carbon - 1) if len(vals_reached) and g is not None else np.nan,
            "seeds_reaching": f"{len(reached)}/{len(seeds)}",
            "OLD_bucket_nsga2": old_bucket,
            "OLD_bucket_pct_vs_greedy": 100 * (old_bucket / g_carbon - 1) if g is not None else np.nan,
            "random_mean": r.mean() if len(r) else np.nan,
        })
        for s in seeds:
            if s in best:
                per_seed.append({"budget": d, "seed": s, "carbon": best[s][0], "disruption_used": best[s][1],
                                 "reached": s in reached,
                                 "pct_vs_greedy": 100 * (best[s][0] / g_carbon - 1) if g is not None else np.nan,
                                 "pct_vs_greedy_fill": 100 * (best[s][0] / gf_carbon - 1) if gf is not None else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(per_seed), seed_max_disruption


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    table, per_seed, seed_max = compare(load_scenarios())

    d3 = lambda x: f"{x:.3f}"            # disruption values shown to 3 decimals: the detail matters here
    print("Max disruption each NSGA-II seed's front reaches:")
    print(seed_max.map(d3).to_string(), "\n")
    print("=== Budget-matched comparison (best carbon WITHIN each disruption budget) ===")
    print("greedy = stops at the first parcel that overshoots (depends on tie order); greedy_fill = skips it and keeps filling.")
    print(table[["budget", "greedy_carbon", "greedy_fill_carbon", "nsga2_best_seed_carbon", "nsga2_best_pct_vs_greedy",
                 "nsga2_best_pct_vs_greedy_fill", "nsga2_mean_reached", "nsga2_sd_reached", "nsga2_mean_pct_vs_greedy_fill",
                 "seeds_reaching", "random_mean"]].to_string(index=False, formatters={"budget": d3}))
    print("\nDisruption actually used:")
    print(table[["budget", "greedy_disruption_used", "greedy_fill_disruption_used", "nsga2_best_disruption_used"]]
          .to_string(index=False, formatters={k: d3 for k in ("budget", "greedy_disruption_used", "greedy_fill_disruption_used", "nsga2_best_disruption_used")}))
    print("\n=== For contrast: the OLD rounded-bucket reading (biased toward NSGA-II) ===")
    print(table[["budget", "OLD_bucket_nsga2", "OLD_bucket_pct_vs_greedy",
                 "nsga2_best_seed_carbon", "nsga2_best_pct_vs_greedy"]].to_string(index=False))
    print("\n=== Per seed (best within budget); 'reached' = front extends to the budget ===")
    print(per_seed.to_string(index=False, formatters={"budget": d3, "disruption_used": d3}))
