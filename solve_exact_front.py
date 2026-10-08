"""
Exact optimum for the reallocation problem NSGA-II is searching.

WHY THIS IS POSSIBLE: the objective is additively separable. Carbon is
sum_p area_p * lookup[p, X_p] (each parcel's contribution depends only on ITS
OWN assigned class), and disruption is the area share of parcels whose class
changes. Spatial adjacency appears nowhere in the objective (the GNN only
seeds the search). For a disruption budget d the problem is therefore a
multiple-choice knapsack:

    maximize    sum_{p,k} gain[p,k] * z[p,k]
    subject to  sum_k z[p,k] <= 1                      one new class per parcel
                sum_{p,k} area[p] * z[p,k] <= d * total_area

solved to proven optimality with a mixed-integer LP (HiGHS, scipy.optimize.milp)
on the SAME lookup tables NSGA-II is scored on (ScenarioProblem._tables).

TWO VERSIONS OF THE PROBLEM, because the methods were not solving the same one.
Some parcels' own class is not among the classes NSGA-II can choose from
(e.g. 'unresolved_crop'): ScenarioProblem gives them baseline_idx = -1, so
evaluate_batch ALWAYS counts them as changed and NSGA-II has no way to leave
them as they are. Greedy and random baselines can leave them untouched.
  * "as searched"      : those parcels MUST be reassigned (and always count as
                         disruption). The exact ceiling for what NSGA-II could
                         ever find.
  * "unchanged allowed": every parcel may stay as it is. The problem greedy and
                         random actually solve; the fair ceiling for them.
The gap between the two shows what the encoding costs.

Usage (from the repo root, after the search and baselines have been exported):
    python solve_exact_front.py          # the six-budget comparison, ~1 min
    python solve_exact_front.py --fine   # also the exact front on a 70-point grid
Writes data/exact_comparison.csv (and data/exact_front.csv with --fine).
"""
import contextlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
FINE_GRID = np.round(np.arange(0.01, 0.701, 0.01), 2)

TOL_REPAIRABLE = 1e-5   # budget overshoot (as a fraction of total area) treated as solver tolerance
TOL_INTEGRAL = 1e-6
MIP_REL_GAP = 1e-4     # solutions certified within 0.01% of the true optimum: far finer than any difference that matters here
TIME_LIMIT_S = 60      # per solve; if hit, the achieved gap is reported


@contextlib.contextmanager
def _silence_c_output():
    """HiGHS prints internal debug lines from C++ straight to the process's
    stdout/stderr, bypassing Python. Results are unaffected; this only keeps
    the console readable. Falls back to a no-op if the descriptors can't be
    redirected."""
    try:
        sys.stdout.flush(); sys.stderr.flush()
        saved = [os.dup(1), os.dup(2)]
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1); os.dup2(devnull, 2)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in saved + [devnull]:
            os.close(fd)


class ExactSolver:
    def __init__(self, problem, evaluate_batch, forced_parcels_must_change=True):
        """forced_parcels_must_change=True  -> "as searched" (matches evaluate_batch exactly)
           forced_parcels_must_change=False -> "unchanged allowed" (matches greedy/random)"""
        self.problem = problem
        self.evaluate_batch = evaluate_batch
        self.forced_must_change = forced_parcels_must_change
        t = problem._tables
        self.area = np.asarray(t["area_ha"], dtype=float)
        self.lookup = np.asarray(t["padded_carbon_lookup"], dtype=float)
        self.base_idx = np.asarray(t["baseline_idx_per_parcel"])
        self.total_area = float(t["total_area_ha"])
        self.n = len(self.area)
        self.sentinel = self.base_idx < 0      # own class not in own domain
        self.repairs = 0
        self.max_initial_excess = 0.0
        self.worst_gap = 0.0
        self.time_limited = 0

        carbon_by_pid = dict(zip(problem.combined["parcel_id"], problem.combined["carbon_t_c_per_ha_mean"]))
        self.base_per_ha = np.array([carbon_by_pid[pid] for pid in problem.parcel_ids], dtype=float)
        ok = ~self.sentinel
        assert np.allclose(self.lookup[np.arange(self.n)[ok], self.base_idx[ok]], self.base_per_ha[ok]), \
            "no-change lookup entries differ from baseline carbon: tables are not what this solver assumes"

        vp, vk, gain = [], [], []
        for p in range(self.n):
            for k in range(int(problem.xu[p]) + 1):
                if k == self.base_idx[p]:
                    continue
                vp.append(p)
                vk.append(k)
                gain.append(self.area[p] * (self.lookup[p, k] - self.base_per_ha[p]))
        self.vp, self.vk, self.gain = np.array(vp), np.array(vk), np.array(gain)
        self.nv = len(self.vp)
        self.A_parcel = csr_matrix((np.ones(self.nv), (self.vp, np.arange(self.nv))), shape=(self.n, self.nv))
        must = self.sentinel & self.forced_must_change
        self.row_lb = np.where(must, 1.0, 0.0)
        self.row_ub = np.ones(self.n)
        self.area_row = csr_matrix(self.area[self.vp][None, :])
        self.forced_disruption = float(self.area[must].sum() / self.total_area)

    def _milp(self, rhs_ha):
        with _silence_c_output():
            return milp(
                c=-self.gain,
                constraints=[LinearConstraint(self.A_parcel, self.row_lb, self.row_ub),
                             LinearConstraint(self.area_row, -np.inf, rhs_ha)],
                integrality=np.ones(self.nv),
                bounds=Bounds(0, 1),
                options={"mip_rel_gap": MIP_REL_GAP, "time_limit": TIME_LIMIT_S},
            )

    def solve(self, budget):
        """Returns dict(carbon_delta_t, disruption, z, X) or None if infeasible.
        X is the class-index vector in the search's own encoding ("as searched"
        version only; None for the 'unchanged allowed' version)."""
        if budget < self.forced_disruption - 1e-12:
            return None
        rhs = budget * self.total_area
        for attempt in range(6):
            res = self._milp(rhs)
            if res.x is None:
                if res.status == 2:                       # proven infeasible
                    return None
                raise AssertionError(f"No feasible solution found at budget {budget} within the {TIME_LIMIT_S}s limit "
                                     f"(status {res.status}: {res.message})")
            if not res.success:                           # time limit reached WITH a feasible solution: usable, gap certified
                self.time_limited += 1
            self.worst_gap = max(self.worst_gap, float(getattr(res, "mip_gap", 0.0) or 0.0))
            z = np.round(res.x).astype(int)
            frac = float(np.max(np.abs(res.x - z)))
            if frac > TOL_INTEGRAL:
                raise AssertionError(f"MILP returned a non-integral solution (max fractional part {frac:.2e}) at budget {budget}")
            disruption = float(self.area[self.vp[z == 1]].sum() / self.total_area)
            excess = disruption - budget
            if excess <= 1e-12:
                break
            self.max_initial_excess = max(self.max_initial_excess, excess)
            if excess > TOL_REPAIRABLE:
                raise AssertionError(
                    f"At budget {budget}: solution uses disruption {disruption:.8f}, over the budget by {excess:.2e} "
                    f"(> {TOL_REPAIRABLE:.0e}, too large to be solver tolerance). MILP status: {res.status} "
                    f"'{res.message}'. This points to a real formulation problem; please report these numbers.")
            # within solver tolerance: tighten the constraint slightly and re-solve so the result is strictly feasible
            rhs -= excess * self.total_area * 2 + 1e-6
            self.repairs += 1
        else:
            raise AssertionError(f"Could not obtain a strictly feasible solution at budget {budget} after repairs")

        carbon = float(self.gain[z == 1].sum())
        chosen = np.flatnonzero(z)
        X = None
        if self.forced_must_change:
            X = np.where(self.sentinel, 0, self.base_idx).astype(int)
            X[self.vp[chosen]] = self.vk[chosen]
            F = self.evaluate_batch(X[None, :], self.problem._tables)[0]
            assert abs(-F[0] - carbon) <= 1e-6 * max(1.0, abs(carbon)) and abs(F[1] - disruption) <= 1e-9, \
                (f"MILP (carbon {carbon:.6f}, disruption {disruption:.9f}) disagrees with evaluate_batch "
                 f"({-F[0]:.6f}, {F[1]:.9f}): formulation drifted from the search objective")
        return {"carbon_delta_t": carbon, "disruption": disruption, "z": z, "X": X}

    def forced_parcel_table(self):
        """Breakdown of the parcels the search can never leave unchanged."""
        if not self.sentinel.any():
            return pd.DataFrame()
        cls = self.problem.combined.set_index("parcel_id").loc[self.problem.parcel_ids, "land_cover_class"].to_numpy()
        rows = []
        best_gain = np.full(self.n, -np.inf)
        np.maximum.at(best_gain, self.vp, self.gain)
        for c in sorted(set(cls[self.sentinel])):
            m = self.sentinel & (cls == c)
            rows.append({"original_class": c, "parcels": int(m.sum()),
                         "area_ha": self.area[m].sum(), "share_of_region_%": 100 * self.area[m].sum() / self.total_area,
                         "mean_baseline_tC_per_ha": float(np.average(self.base_per_ha[m], weights=self.area[m])),
                         "best_case_gain_tC": float(best_gain[m].sum())})
        return pd.DataFrame(rows)


def verify_against_export(problem, evaluate_batch, export_path, rtol=1e-6):
    """Rebuild one exported NSGA-II scenario's class assignment, re-score it with
    THIS problem's tables, and require it to match the carbon/disruption the
    search itself recorded. Proves this solver is working on the same problem
    (same region, same carbon accounting) that produced the exports."""
    ex = pd.read_parquet(export_path)
    sid = sorted(ex["scenario_id"].unique())[0]
    sc = ex[ex["scenario_id"] == sid].set_index("parcel_id")
    assert set(sc.index) == set(problem.parcel_ids), \
        "export covers a different set of parcels than this problem (region or exclusion rules differ)"
    X = np.array([problem.per_parcel_labels[p].index(sc.loc[pid, "assigned_class"])
                  for p, pid in enumerate(problem.parcel_ids)])
    F = evaluate_batch(X[None, :], problem._tables)[0]
    for got, want, name in [(-F[0], sc["carbon_delta_vs_baseline_t"].iloc[0], "carbon"),
                            (F[1], sc["disruption_fraction"].iloc[0], "disruption")]:
        assert abs(got - want) <= rtol * max(1.0, abs(want)), \
            f"{name}: re-scored {got:.6f} vs exported {want:.6f}: this is not the same problem the search ran"
    print(f"Verified: re-scoring exported scenario {sid} reproduces its recorded carbon and disruption.")


def build_problem(parcel_ids_path="data/offaly_subregion_parcel_ids.txt", combined=None):
    """Mirrors the setup in nsga2_scenario_search.run_nsga2_search (minus the GNN).
    `combined`: pass an already-loaded (and possibly modified) frame instead of reading the national file."""
    from generate_scenarios import filter_reallocatable, load_area_and_baseline
    from nsga2_scenario_search import ScenarioProblem

    with open(parcel_ids_path) as f:
        parcel_ids = [int(line.strip()) for line in f if line.strip()]
    if combined is None:
        combined = load_area_and_baseline()
    branch_means = combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].mean().to_dict()
    branch_stds = combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].std().to_dict()
    parcel_ids = filter_reallocatable(parcel_ids, combined)
    carbon_by_pid = dict(zip(combined["parcel_id"], combined["carbon_t_c_per_ha_mean"]))
    parcel_ids = [pid for pid in parcel_ids if not pd.isna(carbon_by_pid.get(pid))]
    region_combined = combined[combined["parcel_id"].isin(parcel_ids)].copy()
    return ScenarioProblem(parcel_ids, region_combined, branch_means, branch_stds)


if __name__ == "__main__":
    import glob
    import time
    from nsga2_scenario_search import evaluate_batch
    from compare_methods_at_budget import compare, load_scenarios

    do_fine = "--fine" in sys.argv
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    problem = build_problem()
    seed_files = sorted(glob.glob("data/scenario_details_seed*.parquet"))
    if seed_files:
        verify_against_export(problem, evaluate_batch, seed_files[0])
    searched = ExactSolver(problem, evaluate_batch, forced_parcels_must_change=True)
    relaxed = ExactSolver(problem, evaluate_batch, forced_parcels_must_change=False)
    print(f"{searched.n} parcels, {searched.nv} decision variables", flush=True)

    ft = searched.forced_parcel_table()
    if len(ft):
        print(f"\nParcels the search can NEVER leave unchanged (their own class is not among its choices): "
              f"{searched.forced_disruption:.4f} of the region's area is always counted as disruption.")
        print(ft.to_string(index=False))
        print("(best_case_gain_tC is the carbon those parcels yield when assigned their best class -- carbon "
              "every NSGA-II scenario collects whether or not it is 'earned' by a real reallocation.)\n", flush=True)

    table, _, _ = compare(load_scenarios())
    rows = []
    print(f"Solving the six budgets (each solve capped at {TIME_LIMIT_S}s):", flush=True)
    for d in BUDGETS:
        t0 = time.time(); a = searched.solve(d)
        t1 = time.time(); b = relaxed.solve(d)
        t2 = time.time()
        print(f"  budget {d:.1f}: as searched {t1 - t0:5.1f}s | unchanged allowed {t2 - t1:5.1f}s", flush=True)
        t = table[table["budget"] == d].iloc[0]
        rows.append({
            "budget": f"{d:.3f}",
            "exact_as_searched": a["carbon_delta_t"] if a else np.nan,
            "exact_unchanged_ok": b["carbon_delta_t"],
            "greedy": t["greedy_carbon"],
            "greedy_%_of_unchanged_ok": 100 * t["greedy_carbon"] / b["carbon_delta_t"],
            "greedy_fill": t.get("greedy_fill_carbon", np.nan),
            "greedy_fill_%_of_unchanged_ok": 100 * t.get("greedy_fill_carbon", np.nan) / b["carbon_delta_t"],
            "nsga2_best": t["nsga2_best_seed_carbon"],
            "nsga2_%_of_as_searched": 100 * t["nsga2_best_seed_carbon"] / a["carbon_delta_t"] if a else np.nan,
            "nsga2_%_of_unchanged_ok": 100 * t["nsga2_best_seed_carbon"] / b["carbon_delta_t"],
        })
    result = pd.DataFrame(rows)
    result.to_csv("data/exact_comparison.csv", index=False)
    print("\n=== Exact optima, and each method as % of the ceiling for the problem it actually solves ===")
    print(result.to_string(index=False))
    gap = max(searched.worst_gap, relaxed.worst_gap)
    print(f"\nAll optima certified within {100 * max(gap, 0):.4f}% of the true optimum"
          + (f"; {searched.time_limited + relaxed.time_limited} solve(s) hit the {TIME_LIMIT_S}s limit." if searched.time_limited + relaxed.time_limited else "."))
    if searched.repairs or relaxed.repairs:
        print(f"Note: {searched.repairs + relaxed.repairs} solve(s) needed a feasibility repair for solver tolerance "
              f"(largest initial overshoot {max(searched.max_initial_excess, relaxed.max_initial_excess):.2e} of total area); "
              "all reported solutions are strictly within budget.")
    print("Wrote data/exact_comparison.csv", flush=True)

    if do_fine:
        fine = []
        for i, d in enumerate(FINE_GRID):
            a, b = searched.solve(float(d)), relaxed.solve(float(d))
            fine.append({"budget": d, "as_searched": a["carbon_delta_t"] if a else np.nan,
                         "unchanged_allowed": b["carbon_delta_t"] if b else np.nan})
            if (i + 1) % 10 == 0:
                print(f"  fine grid {i + 1}/{len(FINE_GRID)}", flush=True)
        pd.DataFrame(fine).to_csv("data/exact_front.csv", index=False)
        print("Wrote data/exact_front.csv")
    else:
        print("(add --fine to also compute the exact front on a 70-point grid -> data/exact_front.csv)")
