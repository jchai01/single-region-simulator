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

REALISTIC_TARGET_CLASSES = ["tillage", "grassland", "forestry"]

TARGET_DISRUPTION_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def random_baseline(parcel_ids, combined, branch_means, branch_stds,
                    disruption_fraction, seed=None):
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

    return score_scenario(parcel_ids, new_classes, combined, branch_means, branch_stds,
                          tillage_yield_mean=0.0, tillage_yield_std=0.0, grassland_index_std=0.0)


def greedy_baseline(parcel_ids, combined, branch_means, branch_stds, disruption_fraction):
    region = combined[combined["parcel_id"].isin(parcel_ids)].copy()

    # For each parcel, find its single BEST realistic target class
    # (highest branch-mean carbon among the 3 realistic options) and
    # the resulting per-ha gain over its current class.
    def best_realistic_target(row):
        current = row["land_cover_class"]
        current_carbon = branch_means.get(current, 0.0)
        gains = {
            cls: branch_means[cls] - current_carbon
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

    return score_scenario(parcel_ids, new_classes, combined, branch_means, branch_stds,
                          tillage_yield_mean=0.0, tillage_yield_std=0.0, grassland_index_std=0.0)


def branch_means_from_loaded(combined: pd.DataFrame) -> dict:
    """Same computation as carbon_transition_delta.py's compute_branch_means(),
    but from an ALREADY-LOADED frame -- avoids re-reading the full
    geometry-bearing GPKG from disk a second time (load_area_and_baseline()
    already read it once)."""
    return combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].mean().to_dict()


def branch_stds_from_loaded(combined: pd.DataFrame) -> dict:
    return combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].std().to_dict()


def run_comparison(parcel_ids, n_random_seeds=5):
    combined = load_area_and_baseline()
    branch_means = branch_means_from_loaded(combined)
    branch_stds = branch_stds_from_loaded(combined)
    parcel_ids = filter_reallocatable(parcel_ids, combined)

    rows = []
    for target_disruption in TARGET_DISRUPTION_LEVELS:
        greedy_result = greedy_baseline(parcel_ids, combined, branch_means, branch_stds, target_disruption)
        rows.append({
            "method": "greedy_heuristic", "target_disruption": target_disruption,
            "actual_disruption": greedy_result["disruption_fraction"],
            "carbon_delta_t": greedy_result["carbon_delta_vs_baseline_t"],
        })

        random_gains = [
            random_baseline(parcel_ids, combined, branch_means, branch_stds,
                            target_disruption, seed=s)["carbon_delta_vs_baseline_t"]
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


if __name__ == "__main__":
    with open("data/offaly_subregion_parcel_ids.txt") as f:
        example_parcel_ids = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(example_parcel_ids)} parcel_ids from the saved case-study region")
    run_comparison(example_parcel_ids)
