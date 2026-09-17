"""
Run this ONCE you have the Offaly sub-region's parcel_ids selected --
measures actual per-evaluation cost at the REAL scale, so you can
extrapolate total runtime before committing to a long NSGA-II run
instead of guessing.
"""
import time
import numpy as np
import pandas as pd

from nsga2_scenario_search import ScenarioProblem
from generate_scenarios import load_area_and_baseline, filter_reallocatable

def benchmark(parcel_ids, n_calls=200):
    combined = load_area_and_baseline()
    branch_means = combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].mean().to_dict()
    branch_stds = combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].std().to_dict()
    parcel_ids = filter_reallocatable(parcel_ids, combined)

    # Same NaN-carbon exclusion as run_nsga2_search() -- without this,
    # ScenarioProblem's guard correctly raises rather than silently
    # proceeding, but that means the benchmark crashes instead of
    # measuring the same parcel set the real search run will actually
    # use. Match that behavior here instead of leaving this inconsistent.
    carbon_by_pid = dict(zip(combined["parcel_id"], combined["carbon_t_c_per_ha_mean"]))
    nan_pids = [pid for pid in parcel_ids if pd.isna(carbon_by_pid.get(pid))]
    if nan_pids:
        rate = len(nan_pids) / len(parcel_ids)
        print(f"Excluded {len(nan_pids)} parcel(s) with no assigned carbon "
              f"baseline ({100*rate:.2f}% of this region) from benchmarking: "
              f"{nan_pids}")
        parcel_ids = [pid for pid in parcel_ids if pid not in set(nan_pids)]

    region_combined = combined[combined["parcel_id"].isin(parcel_ids)].copy()

    problem = ScenarioProblem(parcel_ids, region_combined, branch_means, branch_stds)

    # Random valid assignments, respecting each parcel's own domain size
    X = np.array([
        np.random.randint(0, problem.xu + 1) for _ in range(n_calls)
    ])

    start = time.time()
    out = {}
    problem._evaluate(X, out)
    elapsed = time.time() - start

    per_eval_ms = (elapsed / n_calls) * 1000
    print(f"\n{len(parcel_ids)} parcels, {n_calls} evaluations: {elapsed:.2f}s total, "
          f"{per_eval_ms:.2f} ms/evaluation")

    for pop_size, n_gen in [(40, 50), (100, 100), (200, 150), (400, 200)]:
        total_evals = pop_size * n_gen
        est_seconds = total_evals * per_eval_ms / 1000
        print(f"  pop={pop_size:>4}, gen={n_gen:>4} -> {total_evals:>7,} evals "
              f"-> ~{est_seconds/60:.1f} min (evaluation cost only, excludes GNN seeding + NSGA-II overhead)")

if __name__ == "__main__":
    with open("data/offaly_subregion_parcel_ids.txt") as f:
        example_parcel_ids = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(example_parcel_ids)} parcel_ids from the saved case-study region")
    benchmark(example_parcel_ids)
