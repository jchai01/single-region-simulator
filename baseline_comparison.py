"""
Baseline comparison for NSGA-II land reallocation -- the "why deep
learning here" check a domain-journal reviewer will ask for empirically.

REALISTIC_TARGET_CLASSES: deliberately narrower than the GNN's full
label space (['excluded', 'forestry', 'grassland', 'peatland',
'tillage', 'unresolved_crop']). A landowner can genuinely choose to
convert a parcel between tillage/grassland/forestry via active
management. They CANNOT choose to make a parcel become 'peatland'
(soil-determined, not a management decision), nor does 'excluded' or
'unresolved_crop' correspond to any real land-use action. If the GNN
pipeline has been allowed to propose these as reallocation TARGETS
(not just as the untouched baseline for parcels that already ARE
that class), some of its reported carbon gain may be coming from
physically impossible conversions -- worth checking
ScenarioProblem/sample_reallocation's output for this specifically
before trusting prior NSGA-II-vs-sampling comparisons at face value.

Two baselines, both restricted to REALISTIC_TARGET_CLASSES:
  1. RANDOM -- the floor. Randomly reassign a target fraction of
     eligible area to a random realistic class. If NSGA-II doesn't
     beat this comfortably, nothing else about it matters.
  2. GREEDY -- the real baseline reviewers expect. Rank parcels by
     potential carbon gain per hectare if converted to their single
     BEST realistic target class, convert greedily most-to-least
     disruptive... i.e. most-to-least beneficial, until a disruption
     budget is hit. This is exactly the kind of "simple heuristic"
     classical land-use optimization literature uses (see the
     project's own earlier novelty-check: weighted linear combination
     / greedy allocation rules are the established alternative to
     NSGA-II in this literature).

Both are evaluated at the SAME disruption levels as the existing
NSGA-II Pareto front (0.1 through 0.6) for a direct, matched comparison
-- reuses score_scenario() so the carbon/disruption numbers are
computed identically across all three approaches.
"""

import numpy as np
import pandas as pd

from generate_scenarios import load_area_and_baseline, filter_reallocatable, score_scenario
from carbon_time_dynamics import carbon_at_year, YEARS_TO_POLICY_TARGET

REALISTIC_TARGET_CLASSES = ["tillage", "grassland", "forestry"]

TARGET_DISRUPTION_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def random_baseline(parcel_ids, combined, branch_means, branch_stds,
                    disruption_fraction, seed=None):
    """Returns (score_result, new_classes). new_classes (parcel_id ->
    assigned class, for EVERY parcel in the region, changed or not) is
    the full per-parcel record of what this scenario actually did --
    exported by export_scenario_details() below for spatial display."""
    rng = np.random.default_rng(seed)
    region = combined[combined["parcel_id"].isin(parcel_ids)].copy()

    # Randomly select parcels (by area, up to disruption_fraction of
    # total region area) to reassign -- rest keep their current class.
    region = region.sample(frac=1.0, random_state=seed)  # shuffle
    region["cum_area_frac"] = region["area_ha"].cumsum() / region["area_ha"].sum()
    to_change = region["cum_area_frac"] <= disruption_fraction

    new_classes = {}
    for _, row in region.iterrows():
        pid = row["parcel_id"]
        if to_change.loc[row.name]:
            new_classes[pid] = rng.choice(REALISTIC_TARGET_CLASSES)
        else:
            new_classes[pid] = row["land_cover_class"]

    result = score_scenario(parcel_ids, new_classes, combined, branch_means, branch_stds,
                            tillage_yield_mean=0.0, tillage_yield_std=0.0, grassland_index_std=0.0)
    return result, new_classes


def greedy_baseline(parcel_ids, combined, branch_means, branch_stds, disruption_fraction):
    """Returns (score_result, new_classes) -- see random_baseline()'s
    docstring for what new_classes is and why it's returned."""
    region = combined[combined["parcel_id"].isin(parcel_ids)].copy()

    # For each parcel, find its single BEST realistic target class and
    # the resulting per-ha gain, using the SAME time-resolved objective
    # NSGA-II is scored on (carbon_at_year, not the raw branch-mean
    # gap) -- otherwise greedy would be ranking its choices by one
    # objective (eventual gain) while being scored on another (gain
    # realized by YEARS_TO_POLICY_TARGET), understating the strongest
    # fair version of this baseline.
    def best_realistic_target(row):
        current = row["land_cover_class"]
        own_baseline = row["carbon_t_c_per_ha_mean"]
        gains = {
            cls: carbon_at_year(current, cls, own_baseline, branch_means[cls], YEARS_TO_POLICY_TARGET) - own_baseline
            for cls in REALISTIC_TARGET_CLASSES if cls != current
        }
        if not gains:
            return current, 0.0
        best_cls = max(gains, key=gains.get)
        return best_cls, gains[best_cls]

    region[["best_target", "gain_per_ha"]] = region.apply(
        lambda r: pd.Series(best_realistic_target(r)), axis=1
    )

    # Greedy: convert parcels in order of DESCENDING gain_per_ha until
    # the disruption budget (fraction of total region area) is hit.
    region = region.sort_values("gain_per_ha", ascending=False)
    region["cum_area_frac"] = region["area_ha"].cumsum() / region["area_ha"].sum()

    new_classes = {}
    for _, row in region.iterrows():
        pid = row["parcel_id"]
        if row["cum_area_frac"] <= disruption_fraction and row["gain_per_ha"] > 0:
            new_classes[pid] = row["best_target"]
        else:
            new_classes[pid] = row["land_cover_class"]

    result = score_scenario(parcel_ids, new_classes, combined, branch_means, branch_stds,
                            tillage_yield_mean=0.0, tillage_yield_std=0.0, grassland_index_std=0.0)
    return result, new_classes


def branch_means_from_loaded(combined: pd.DataFrame) -> dict:
    """Same computation as carbon_transition_delta.py's compute_branch_means(),
    but from an ALREADY-LOADED frame -- avoids re-reading the full
    geometry-bearing GPKG from disk a second time (load_area_and_baseline()
    already read it once)."""
    return combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].mean().to_dict()


def branch_stds_from_loaded(combined: pd.DataFrame) -> dict:
    return combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].std().to_dict()


def run_comparison(parcel_ids, n_random_seeds=5, combined=None, branch_means=None,
                   branch_stds=None, skip_filter=False):
    """
    combined/branch_means/branch_stds: pass these in (already loaded) to
    avoid re-reading the geopackage and recomputing branch stats when
    calling this alongside export_scenario_details() on the same data --
    see __main__ below. Defaults to loading them internally (unchanged
    behavior) if not provided, so standalone calls elsewhere still work.

    skip_filter: set True if parcel_ids has already been through
    filter_reallocatable() by the caller, to avoid the exclusion message
    printing a second time.
    """
    if combined is None:
        combined = load_area_and_baseline()
    if branch_means is None:
        branch_means = branch_means_from_loaded(combined)
    if branch_stds is None:
        branch_stds = branch_stds_from_loaded(combined)
    if not skip_filter:
        parcel_ids = filter_reallocatable(parcel_ids, combined)

    rows = []
    for target_disruption in TARGET_DISRUPTION_LEVELS:
        greedy_result, _ = greedy_baseline(parcel_ids, combined, branch_means, branch_stds, target_disruption)
        rows.append({
            "method": "greedy_heuristic", "target_disruption": target_disruption,
            "actual_disruption": greedy_result["disruption_fraction"],
            "carbon_delta_t": greedy_result["carbon_delta_vs_baseline_t"],
        })

        random_gains = [
            random_baseline(parcel_ids, combined, branch_means, branch_stds,
                            target_disruption, seed=s)[0]["carbon_delta_vs_baseline_t"]
            for s in range(n_random_seeds)
        ]
        rows.append({
            "method": "random_baseline", "target_disruption": target_disruption,
            "actual_disruption": target_disruption,  # approx, random selection targets this directly
            "carbon_delta_t": float(np.mean(random_gains)),
            "carbon_delta_t_std_across_seeds": float(np.std(random_gains)),
        })

    result = pd.DataFrame(rows)
    print(result.round(1).to_string(index=False))
    return result


def export_scenario_details(parcel_ids, combined, branch_means, branch_stds,
                            n_random_seeds=5, skip_filter=False,
                            out_path="data/scenario_details_baselines.parquet"):
    """
    Long-format export matching nsga2_scenario_search.py's
    export_scenario_details() schema exactly: one row per (method,
    scenario_id, parcel_id), columns method, scenario_id,
    disruption_fraction, carbon_delta_vs_baseline_t, parcel_id,
    original_class, assigned_class, changed.

    method="greedy": one scenario_id per disruption level (the
    disruption level itself, as a string, e.g. "0.4").
    method="random": one scenario_id per (disruption level, seed) pair
    (e.g. "0.4_seed2") -- unlike run_comparison()'s aggregate table,
    which averages random's carbon gain across seeds, this export keeps
    each seed's own scenario separately displayable, since "the random
    baseline" isn't really one scenario but a distribution of them.

    Applies filter_reallocatable() internally by default, so callers can
    pass the raw region parcel_ids without pre-filtering. Set
    skip_filter=True if the caller already filtered (e.g. __main__ below,
    sharing one filtered parcel_ids list with run_comparison()) to avoid
    the exclusion message printing a second time.
    """
    if not skip_filter:
        parcel_ids = filter_reallocatable(parcel_ids, combined)
    region = combined[combined["parcel_id"].isin(parcel_ids)]
    original_class_by_pid = dict(zip(region["parcel_id"], region["land_cover_class"]))

    rows = []

    for target_disruption in TARGET_DISRUPTION_LEVELS:
        greedy_result, greedy_new_classes = greedy_baseline(
            parcel_ids, combined, branch_means, branch_stds, target_disruption,
        )
        scenario_id = f"{target_disruption:.1f}"
        for pid in parcel_ids:
            original = original_class_by_pid[pid]
            assigned = greedy_new_classes[pid]
            rows.append({
                "method": "greedy", "scenario_id": scenario_id,
                "disruption_fraction": greedy_result["disruption_fraction"],
                "carbon_delta_vs_baseline_t": greedy_result["carbon_delta_vs_baseline_t"],
                "parcel_id": pid, "original_class": original, "assigned_class": assigned,
                "changed": assigned != original,
            })

        for s in range(n_random_seeds):
            random_result, random_new_classes = random_baseline(
                parcel_ids, combined, branch_means, branch_stds, target_disruption, seed=s,
            )
            scenario_id = f"{target_disruption:.1f}_seed{s}"
            for pid in parcel_ids:
                original = original_class_by_pid[pid]
                assigned = random_new_classes[pid]
                rows.append({
                    "method": "random", "scenario_id": scenario_id,
                    "disruption_fraction": random_result["disruption_fraction"],
                    "carbon_delta_vs_baseline_t": random_result["carbon_delta_vs_baseline_t"],
                    "parcel_id": pid, "original_class": original, "assigned_class": assigned,
                    "changed": assigned != original,
                })

    details = pd.DataFrame(rows)
    details.to_parquet(out_path, index=False)
    print(f"Exported per-parcel scenario detail for greedy ({len(TARGET_DISRUPTION_LEVELS)} scenarios) "
          f"and random ({len(TARGET_DISRUPTION_LEVELS) * n_random_seeds} scenarios) "
          f"({len(details)} rows) to {out_path}")
    return details


if __name__ == "__main__":
    with open("data/offaly_subregion_parcel_ids.txt") as f:
        example_parcel_ids = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(example_parcel_ids)} parcel_ids from the saved case-study region")

    # Load once, share across both calls below -- avoids reading the
    # geopackage and filtering parcel_ids twice (and printing the
    # exclusion message twice).
    _combined = load_area_and_baseline()
    _branch_means = branch_means_from_loaded(_combined)
    _branch_stds = branch_stds_from_loaded(_combined)
    _filtered_parcel_ids = filter_reallocatable(example_parcel_ids, _combined)

    run_comparison(_filtered_parcel_ids, combined=_combined, branch_means=_branch_means,
                   branch_stds=_branch_stds, skip_filter=True)
    export_scenario_details(_filtered_parcel_ids, _combined, _branch_means, _branch_stds,
                            skip_filter=True)
