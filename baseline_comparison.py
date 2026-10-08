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
from carbon_time_dynamics import policy_year_stock, YEARS_TO_POLICY_TARGET

REALISTIC_TARGET_CLASSES = ["tillage", "grassland", "forestry"]

TARGET_DISRUPTION_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def best_realistic_target(current_class, own_baseline, branch_means):
    """Greedy's per-parcel choice: the realistic target class with the highest
    per-hectare gain (t C/ha at YEARS_TO_POLICY_TARGET), and that gain.

    Module-level, and built on carbon_time_dynamics.policy_year_stock -- the
    single definition of what a move earns, shared with score_scenario() --
    so test_scoring_consistency.py can verify it against the scorer and the
    NSGA-II lookup table. (It used to be an inner function with its own copy
    of the formula; when the SOLUM hybrid was wired into the scorer but not
    into this copy, greedy ranked parcels almost in reverse of what the
    scorer rewards.)"""
    gains = {
        cls: policy_year_stock(current_class, cls, own_baseline, branch_means) - own_baseline
        for cls in REALISTIC_TARGET_CLASSES if cls != current_class
    }
    if not gains:
        return current_class, 0.0
    best_cls = max(gains, key=gains.get)
    return best_cls, gains[best_cls]


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
            # A "reallocation" must actually change the class: drawing from all
            # three realistic classes let a parcel be "reallocated" to its own
            # class, which used up budget without changing anything.
            options = [c for c in REALISTIC_TARGET_CLASSES if c != row["land_cover_class"]]
            new_classes[pid] = rng.choice(options)
        else:
            new_classes[pid] = row["land_cover_class"]

    result = score_scenario(parcel_ids, new_classes, combined, branch_means, branch_stds,
                            tillage_yield_mean=0.0, tillage_yield_std=0.0, grassland_index_std=0.0)
    return result, new_classes


def greedy_baseline(parcel_ids, combined, branch_means, branch_stds, disruption_fraction, fill=False):
    """Returns (score_result, new_classes) -- see random_baseline()'s
    docstring for what new_classes is and why it's returned.

    Two selection rules over the same density ranking:
      fill=False ("stop"): convert parcels in descending gain per ha and STOP at the first one that
          would overshoot the budget. Its result depends on how EXACT ties are ordered (on the
          case-study region the spread over tie orders was up to 9% at the 0.1 budget), so it is
          reported with its tie-break, never alone.
      fill=True: the same, but SKIP a parcel that does not fit and keep scanning, so small parcels
          fill the remaining budget. Far less sensitive to tie order and within a fraction of a
          percent of the exact optimum on the case-study region: the strong baseline."""
    region = combined[combined["parcel_id"].isin(parcel_ids)].copy()

    # Ranking uses best_realistic_target() below, which calls policy_year_stock():
    # the SAME function score_scenario() scores with, so ranking and scoring cannot disagree.
    region[["best_target", "gain_per_ha"]] = region.apply(
        lambda r: pd.Series(best_realistic_target(r["land_cover_class"], r["carbon_t_c_per_ha_mean"], branch_means)), axis=1
    )

    # Greedy: convert parcels in order of DESCENDING gain_per_ha until
    # the disruption budget (fraction of total region area) is hit.
    # Many parcels have EXACTLY equal gain per ha (carbon baselines come from soil-association
    # values, so parcels of one association and class tie). Sorting by gain alone leaves the
    # order among ties arbitrary, and the budget boundary then falls on a different parcel
    # depending on the sort implementation: the same rule gave results differing by up to ~1%.
    # Ties are broken by parcel_id (ascending), on a key rounded to 1e-9 t C/ha so float noise
    # cannot reorder them. mckp_tools.solve_greedy uses the identical rule.
    region["_density_key"] = region["gain_per_ha"].round(9)
    region = region.sort_values(["_density_key", "parcel_id"], ascending=[False, True], kind="mergesort")
    cap = disruption_fraction * float(region["area_ha"].sum())      # hectares that may change
    used, stopped = 0.0, False
    new_classes = {}
    for _, row in region.iterrows():
        pid = row["parcel_id"]
        if row["gain_per_ha"] > 0 and not stopped:
            if used + row["area_ha"] <= cap + 1e-12:
                new_classes[pid] = row["best_target"]
                used += row["area_ha"]
                continue
            if not fill:
                stopped = True            # the "stop" rule: the first parcel that does not fit ends the conversion
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
        fill_result, _ = greedy_baseline(parcel_ids, combined, branch_means, branch_stds, target_disruption, fill=True)
        rows.append({
            "method": "greedy_fill_heuristic", "target_disruption": target_disruption,
            "actual_disruption": fill_result["disruption_fraction"],
            "carbon_delta_t": fill_result["carbon_delta_vs_baseline_t"],
        })

        random_results = [
            random_baseline(parcel_ids, combined, branch_means, branch_stds,
                            target_disruption, seed=s)[0]
            for s in range(n_random_seeds)
        ]
        random_gains = [r["carbon_delta_vs_baseline_t"] for r in random_results]
        rows.append({
            "method": "random_baseline", "target_disruption": target_disruption,
            "actual_disruption": float(np.mean([r["disruption_fraction"] for r in random_results])),
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
        scenario_id = f"{target_disruption:.1f}"
        for method, fill in (("greedy", False), ("greedy_fill", True)):
            greedy_result, greedy_new_classes = greedy_baseline(
                parcel_ids, combined, branch_means, branch_stds, target_disruption, fill=fill,
            )
            for pid in parcel_ids:
                original = original_class_by_pid[pid]
                assigned = greedy_new_classes[pid]
                rows.append({
                    "method": method, "scenario_id": scenario_id,
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
    print(f"Exported per-parcel scenario detail for greedy and greedy_fill ({len(TARGET_DISRUPTION_LEVELS)} scenarios each) "
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
