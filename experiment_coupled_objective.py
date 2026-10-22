"""
Where do search methods start to matter?  Additive carbon objective + a boundary penalty.

    J = carbon gain (t C) - lam * (neighbouring parcel pairs in different final classes)

Compared at each (budget, lam):
  greedy (paper)       fill-greedy by carbon per hectare, blind to the boundary term
  greedy + polish      the same, then single-parcel local search on the exact J
  aware greedy         ranks by (gain - lam * change in boundary) per hectare
  aware + polish       aware greedy, then local search
  NSGA-II (3 obj)      carbon, disruption, boundary; best J among EVERY solution it evaluated within budget (an archive, so crowding-distance
                       truncation cannot throw the best one away) (--nsga);
                       --nsga-seeded adds the fair-start control (initial population includes the greedy solutions);
                       use --seeds 1 2 3 4 and larger --pop/--generations to give it every chance
  exact MILP           with a boundary variable per edge. Reports its proven gap; if it stops at the
                       time limit, the table says so and uses the solver's DUAL BOUND, so every
                       "gap to optimum" column is an honest upper bound on the distance to optimal.

lam is expressed as a multiple (lam_f) of the mean gain of a gain-positive conversion, i.e. "how many
average conversions is one extra boundary edge worth". lam_f = 0 reproduces the additive paper result.

Real data (from the repository root, same data as the other experiments):
    python experiment_coupled_objective.py --nsga --nsga-seeded --seeds 1 2 3 4 --pop 300 --generations 1000
Plumbing check without data (NOT evidence about Ireland):
    python experiment_coupled_objective.py --synthetic 150 --time-limit 30
Writes data/coupled_objective.csv.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

BUDGETS = [0.1, 0.3, 0.5]
LAM_F = [0.0, 0.1, 0.3, 1.0, 3.0]


def synthetic_world(n, seed=0, k=3):
    import test_scoring_consistency as tsc
    import nsga2_scenario_search as nss
    c, ids, m, s = tsc.synthetic_region(n=n, seed=seed)
    problem = nss.ScenarioProblem(ids, c, m, s)
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2))
    d = ((xy[:, None] - xy[None]) ** 2).sum(-1); np.fill_diagonal(d, np.inf)
    pairs = {(min(i, j), max(i, j)) for i in range(n) for j in np.argsort(d[i])[:k]}
    return problem, sorted(pairs)


def real_world():
    import solve_exact_front as sef
    from mckp_tools import edges_to_index_pairs
    problem = sef.build_problem()
    edges = pd.read_parquet("data/graph_edges.parquet")
    return problem, edges_to_index_pairs(edges, problem.parcel_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=0, help="use a synthetic region of this many parcels")
    ap.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    ap.add_argument("--lam-f", type=float, nargs="+", default=LAM_F)
    ap.add_argument("--time-limit", type=float, default=300, help="seconds per MILP solve")
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--nsga", action="store_true", help="also run 3-objective NSGA-II (random start)")
    ap.add_argument("--nsga-seeded", action="store_true",
                    help="also run NSGA-II started from the aware-greedy+polish solutions of every (budget, lam) "
                         "cell, the fair-start control; implies --nsga")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--out", default="data/coupled_objective.csv")
    a = ap.parse_args()

    import nsga2_scenario_search as nss
    import solve_exact_front as sef
    import coupled_objective as co

    problem, pairs = synthetic_world(a.synthetic) if a.synthetic else real_world()
    solver = sef.ExactSolver(problem, nss.evaluate_batch, False)
    cp = co.CoupledProblem(solver, pairs)
    pos = cp.mv_gain[cp.mv_gain > 0]
    unit = float(pos.mean())
    deg = np.array([len(x) for x in cp.nbrs])
    print(f"{cp.n} parcels, {cp.ne:,} adjacencies (mean degree {deg.mean():.1f}), baseline boundary {cp.base_cut:,.0f}; "
          f"unit = mean positive gain = {unit:,.1f} t C", flush=True)
    if a.synthetic:
        print("SYNTHETIC REGION: plumbing check only, not evidence about the real data.", flush=True)

    # NSGA-II is lam-independent (three objectives), so run it once per seed and variant and reuse
    cells = [(b, lf * unit) for b in a.budgets for lf in a.lam_f]
    nsga = []                                   # (label, problem with archive)
    if a.nsga or a.nsga_seeded:
        variants = [("plain", ())]
        if a.nsga_seeded:
            seed_sols = [cp.local_search(cp.greedy_aware(b, lf * unit), b, lf * unit)
                         for b in a.budgets for lf in a.lam_f]
            variants.append(("seeded", seed_sols))
        for vname, sols0 in variants:
            for sd in a.seeds:
                t0 = time.time()
                F, X, prob = co.run_nsga2_3obj(cp, n_generations=a.generations, pop_size=a.pop, seed=sd,
                                               seed_solutions=sols0, cells=cells)
                nsga.append((f"NSGA-II {vname} seed {sd}", prob))
                print(f"NSGA-II {vname} seed {sd}: {len(F)} non-dominated solutions in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for b in a.budgets:
        for lf in a.lam_f:
            lam = lf * unit
            t0 = time.time()
            ch_x, info = cp.solve_exact(b, lam, gap=a.gap, time_limit=a.time_limit)
            sols = {"exact": ch_x}
            g0 = cp.greedy_naive(b)
            sols["greedy (paper)"] = g0
            sols["greedy + polish"] = cp.local_search(g0, b, lam)
            g1 = cp.greedy_aware(b, lam)
            sols["aware greedy"] = g1
            sols["aware + polish"] = cp.local_search(g1, b, lam)
            for nlabel, nprob in nsga:
                J, x = nprob.archive[cells.index((b, lam))]
                if x is None:
                    continue
                base_idx = np.asarray(nprob._tables["baseline_idx_per_parcel"])
                choice = np.full(cp.n, -1)
                for p in np.flatnonzero(x != base_idx):
                    choice[p] = next(m for m in cp.moves_of[p] if cp.mv_k[m] == x[p])
                sols[nlabel] = choice
            ub = info["dual_bound_J"] if not info["proven"] else info["J"]
            for name, ch in sols.items():
                assert cp.feasible(ch, b), f"{name} violates the budget"
                ev = cp.evaluate(ch, lam)
                rows.append({"budget": b, "lam_f": lf, "lam_tC_per_edge": lam, "method": name, **ev,
                             "J_upper_bound": ub, "gap_to_bound_pct": 100 * (ub - ev["J"]) / max(abs(ub), 1e-12),
                             "milp_proven": info["proven"], "milp_gap_pct": 100 * info["gap"],
                             "milp_seconds": info["seconds"]})
            print(f"budget {b:g}  lam_f {lf:g}: MILP {'proven' if info['proven'] else 'TIME LIMIT'} "
                  f"(gap {100*info['gap']:.3g}%, {info['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("\n=== Gap to the exact optimum / best bound (%) of the coupled objective J ===")
    t = df.pivot_table(index=["budget", "lam_f"], columns="method", values="gap_to_bound_pct")
    print(t.to_string())
    print("\n=== Boundary edges and patches of each method's solution ===")
    print(df.pivot_table(index=["budget", "lam_f"], columns="method", values="boundary").to_string())
    print(df.pivot_table(index=["budget", "lam_f"], columns="method", values="patches").to_string())
    bad = df[~df["milp_proven"]][["budget", "lam_f"]].drop_duplicates()
    if len(bad):
        print("\nMILP hit the time limit for:", bad.values.tolist(),
              "-- gaps there are measured against the solver's dual bound (an upper bound on the distance to optimal).")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
