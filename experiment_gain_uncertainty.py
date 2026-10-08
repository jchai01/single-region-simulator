"""
Experiment 2a -- does "greedy is near-optimal" survive uncertainty in the carbon gains?

The headline result (greedy reaches 99.1-100% of the provable optimum) is computed on
POINT ESTIMATES of every parcel's gain. Two questions a reviewer will ask:

 (A) STRUCTURAL: if the true gains differ from our estimates, is greedy still
     near-optimal on the true gains? Each draw perturbs the gains, then solves the
     perturbed problem exactly and with greedy. If the ratio stays near 100% for
     every draw, near-optimality is a property of the problem class, not a fluke of
     one set of numbers.
 (B) DECISION VALUE: plans are made on the point estimates but the world follows the
     perturbed gains. How much does even the EXACT-optimal plan lose against a plan
     made with perfect knowledge ("regret from uncertainty"), and how does that compare
     with the difference between the exact plan and the greedy plan? If the
     optimiser-vs-greedy difference is a small fraction of the uncertainty regret, no
     amount of optimisation effort can be justified by these data.

Noise is mean-preserving lognormal: factor = exp(sigma*z - sigma^2/2), z ~ N(0,1).
  --mode parcel (default): one factor per parcel, applied to all its moves. This is what an
        error in the parcel's baseline carbon stock does (gains scale with the baseline).
  --mode move: independent factor per (parcel, class) move. A harsher test of the ranking.
Noise is independent between parcels; real errors are spatially correlated (shared soil
association), which this does not model.

Usage (repo root):
    python experiment_gain_uncertainty.py --sigma-col carbon_t_c_per_ha_std   # the data's own per-parcel uncertainty (main result)
    python experiment_gain_uncertainty.py --sigmas 0.1 0.2 0.3                # generic stress test
  options: --draws, --mode parcel|move, --greedy-rule fill|stop, --tag
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from mckp_tools import parcel_instance, solve_exact, solve_greedy

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def noise_factors(inst, sigma, rng, mode):
    """Mean-preserving lognormal factors. `sigma` is a scalar, or an array with one sigma per group
    (parcel) for heteroscedastic noise, e.g. from the data's own per-parcel uncertainty."""
    s = np.asarray(sigma, dtype=float)
    if s.ndim == 0 and float(s) == 0:
        return np.ones(len(inst.gain))
    if mode == "parcel":
        sg = s if s.ndim == 1 else np.full(inst.n_groups, float(s))
        per_group = np.exp(sg * rng.standard_normal(inst.n_groups) - sg ** 2 / 2)
        return per_group[inst.group]
    sm = s[inst.group] if s.ndim == 1 else float(s)
    return np.exp(sm * rng.standard_normal(len(inst.gain)) - np.square(sm) / 2)


def run(inst, sigmas, draws, budgets=BUDGETS, mode="parcel", seed=1, progress=True, greedy_fill=True):
    rng = np.random.default_rng(seed)
    # greedy_fill=True (default): the skip-and-fill greedy, the strong baseline, whose result barely depends on how
    # exactly tied parcels are ordered. False: the stop-at-first-overshoot greedy, whose result moves up to ~10% with tie order.
    plans = {b: (solve_exact(inst, b)[0], solve_greedy(inst, b, fill=greedy_fill)[0]) for b in budgets}   # made on point estimates
    rows = []
    t0 = time.time()
    for item in sigmas:
        label, sig = item if isinstance(item, tuple) else (item, item)     # (label, per-parcel sigma array) or a scalar
        for d in range(draws):
            g2 = inst.gain * noise_factors(inst, sig, rng, mode)
            inst2 = inst.with_gain(g2)
            for b in budgets:
                _, opt2 = solve_exact(inst2, b)
                _, gre2 = solve_greedy(inst2, b, fill=greedy_fill)
                ex_plan, gr_plan = plans[b]
                rows.append({"sigma": label, "draw": d, "budget": b, "opt_true": opt2, "greedy_informed": gre2,
                             "exact_plan_realised": float(g2[ex_plan].sum()), "greedy_plan_realised": float(g2[gr_plan].sum())})
        if progress:
            print(f"  sigma={label:g} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarise(df):
    df = df.copy()
    ok = df["opt_true"] > 0
    df = df[ok]
    df["greedy_informed_pct"] = 100 * df["greedy_informed"] / df["opt_true"]
    df["exact_plan_pct"] = 100 * df["exact_plan_realised"] / df["opt_true"]
    df["greedy_plan_pct"] = 100 * df["greedy_plan_realised"] / df["opt_true"]
    df["plan_diff_pp"] = df["exact_plan_pct"] - df["greedy_plan_pct"]
    g = df.groupby(["sigma", "budget"])
    out = pd.DataFrame({
        "A_greedy_%_of_opt_mean": g["greedy_informed_pct"].mean(),
        "A_greedy_%_of_opt_p5": g["greedy_informed_pct"].quantile(0.05),
        "A_greedy_%_of_opt_worst": g["greedy_informed_pct"].min(),
        "B_exact_plan_realises_%": g["exact_plan_pct"].mean(),
        "B_greedy_plan_realises_%": g["greedy_plan_pct"].mean(),
        "B_exact_minus_greedy_pp": g["plan_diff_pp"].mean(),
        "B_uncertainty_regret_pp": 100 - g["exact_plan_pct"].mean(),
    }).reset_index()
    return df, out


def plot(df, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("(matplotlib unavailable; skipping figure)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for sigma, grp in df.groupby("sigma"):
        m = grp.groupby("budget")["greedy_informed_pct"]
        ax[0].plot(m.mean().index, m.mean().values, marker="o", label=f"sigma={sigma:g}")
        ax[0].fill_between(m.mean().index, m.min().values, m.mean().values, alpha=0.12)
    ax[0].axhline(100, color="grey", lw=0.8)
    ax[0].set_xlabel("disruption budget"); ax[0].set_ylabel("greedy as % of the optimum", fontsize=9)
    ax[0].set_title("(A) greedy vs the optimum, gains perturbed\n(line = mean over draws, shading = down to worst draw)", fontsize=10)
    ax[0].legend(fontsize=8)
    reg = df.groupby("sigma").agg(exact=("exact_plan_pct", lambda s: 100 - s.mean()), greedy=("greedy_plan_pct", lambda s: 100 - s.mean()))
    ax[1].plot(reg.index, reg["exact"], marker="o", label="exact-optimal plan")
    ax[1].plot(reg.index, reg["greedy"], marker="s", ls="--", label="greedy plan")
    ax[1].set_xlabel("uncertainty in the gains (sigma)"); ax[1].set_ylabel("loss vs a perfectly informed plan (%)", fontsize=9)
    ax[1].set_title("(B) loss from planning on point estimates\n(plan made on estimates, judged on perturbed truth)", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--mode", choices=["parcel", "move"], default="parcel")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma-col", default=None, help="use this per-parcel uncertainty column instead of --sigmas")
    ap.add_argument("--sigma-kind", choices=["std", "cv"], default="std",
                    help="--sigma-col holds a standard deviation in t C/ha (divided by the parcel's mean) or a relative spread (used as is)")
    ap.add_argument("--greedy-rule", choices=["fill", "stop"], default="fill",
                    help="fill (default): skip-and-fill greedy, the strong baseline; stop: stops at the first overshoot (tie-order dependent)")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output files; default 'data' with --sigma-col, else 'generic' (so the two runs do not overwrite each other)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    tag = a.tag or ("data" if a.sigma_col else "generic")

    import solve_exact_front as sef
    from nsga2_scenario_search import evaluate_batch

    problem = sef.build_problem()
    solver = sef.ExactSolver(problem, evaluate_batch, True)
    reg = problem.combined.set_index("parcel_id").loc[problem.parcel_ids]
    print(f"{solver.n} candidate parcels.")
    cand = [c for c in reg.columns if any(t in c.lower() for t in ("std", "sd", "uncert", "sigma", "stderr", "_ci"))
            and reg[c].dtype.kind in "fi"]
    if cand:
        print("Columns that may hold a per-parcel uncertainty (use one with --sigma-col):")
        for c in cand:
            v = reg[c].astype(float)
            ratio = (v / reg["carbon_t_c_per_ha_mean"]).replace([np.inf, -np.inf], np.nan)
            print(f"   {c}: median {v.median():.3g}; as a share of the parcel's carbon: median {ratio.median():.1%}, "
                  f"10th-90th percentile {ratio.quantile(.1):.1%}-{ratio.quantile(.9):.1%}")
    else:
        print("No per-parcel uncertainty column found in the region data, so --sigmas are generic assumptions.")
    print("(The spread of carbon between parcels of one class is real heterogeneity, already in each parcel's own baseline; "
          "it is NOT the error of a parcel's estimate, so it is not a guide to sigma.)")
    sigmas = a.sigmas
    if a.sigma_col:
        v = reg[a.sigma_col].astype(float)
        cv = v if a.sigma_kind == "cv" else v / reg["carbon_t_c_per_ha_mean"]
        cv = cv.replace([np.inf, -np.inf], np.nan)
        cv = cv.fillna(cv.median()).clip(0.01, 1.5).to_numpy()
        sigmas = [(float(np.median(cv)), cv)]
        print(f"Using per-parcel sigma from '{a.sigma_col}': median {np.median(cv):.1%}, range {cv.min():.1%}-{cv.max():.1%}.")
    inst = parcel_instance(solver)
    print(f"Running {a.draws} draws x {len(sigmas)} sigma x {len(BUDGETS)} budgets (mode={a.mode}, greedy rule={a.greedy_rule}) ...", flush=True)
    raw = run(inst, sigmas, a.draws, mode=a.mode, seed=a.seed, greedy_fill=(a.greedy_rule == "fill"))
    df, table = summarise(raw)
    os.makedirs("data", exist_ok=True)
    raw.to_csv(f"data/gain_uncertainty_{tag}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"\n=== (A) greedy ({a.greedy_rule} rule) vs the optimum when the TRUE gains differ from the estimates; (B) value of planning under uncertainty ===")
    print(table.to_string(index=False))
    print("\nReading it: the A_ columns ask whether greedy stays near the optimum when the true gains differ from the "
          "estimates (near 100 for the worst draw too = structural). The B_ columns compare 'B_exact_minus_greedy_pp' "
          "(what optimising buys over greedy) with 'B_uncertainty_regret_pp' (what uncertainty costs even the best plan). "
          "Compare them per row: where the regret is much larger, optimiser-vs-greedy differences cannot be resolved given the uncertainty.")
    print(f"Wrote data/gain_uncertainty_{tag}.csv")
    if not a.no_plot:
        plot(df, f"figures/gain_uncertainty_{tag}.png")


if __name__ == "__main__":
    main()
