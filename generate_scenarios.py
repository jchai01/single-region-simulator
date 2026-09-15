"""
Scenario generator/scorer v1 -- the multi-objective "optimizer" for
LandFuture DSS. Not a true NSGA-II implementation yet: generates
several candidate reallocations via the trained GNN's sampling,
scores each on real, honestly-separated metrics (no invented combined
score), and reports them as a tradeoff table for a human to compare --
matching the "Option 2: scenario comparison" framing decided early in
this project, not a single forced-optimal blueprint.

METRICS REPORTED PER SCENARIO (deliberately NOT combined into one
number, since tillage yield (t/ha) and grassland productivity (a
unitless index) have no honest common unit):
  - total_carbon_t: area-weighted, t C, well-defined across ALL
    classes (uses branch means from carbon_transition_delta.py)
  - carbon_delta_vs_baseline_t: change vs. the region's ORIGINAL
    (unmasked) land cover
  - area_ha by resulting class (tillage/grassland/forestry/peatland/excluded)
  - tillage_yield_estimate_t: area-weighted tillage branch-mean yield,
    ONLY meaningful for the tillage portion
  - grassland_productivity_index_mean: area-weighted average of the
    (unitless) grassland index, ONLY meaningful for the grassland portion

KNOWN GAP THIS SCRIPT FIXES: build_parcel_graph.py's node table dropped
geometry/area entirely. This script re-joins area from the combined
parcel_yield_carbon.gpkg via POSITIONAL alignment (same justification
as pair_yield_carbon.py -- both derive from the same row-order-
preserving pipeline). Consider adding area as a proper node feature
upstream instead of re-joining it here, next time the graph is rebuilt.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import torch

from sample_reallocation import load_model_and_graph, sample_reallocation
from carbon_transition_delta import compute_branch_means, compute_branch_stds, classify_land_cover

COMBINED_PATH = "data/parcel_yield_carbon.gpkg"

# Genuinely immutable infrastructure -- physically cannot become
# forest/farmland/peatland without demolition, and should NEVER be
# offered as a reallocation candidate. Deliberately narrower than the
# yield model's broader "excluded" bucket (which also contains
# scheme land like fallow/riparian buffers and ambiguous crops like
# maize -- those are real land uses that COULD be reallocated for
# carbon purposes, just weren't modeled for yield). Conflating the two
# was confirmed to let the optimizer propose converting buildings/roads
# to forest and vice versa, which is physically nonsensical.
IMMUTABLE_CLASSES = {
    "BUILDING", "FARMYARD", "FARM ROAD", "ACCESS ROAD / ROADWAYS",
    "QUARRY", "LAKE / WATERWAY / POND", "GARDENS", "RECREATIONAL AREA",
    "NURSERY",
}

# Area-weighted branch-mean yield for tillage only (t/ha) -- computed
# live from the combined data, same "no hardcoded numbers" principle
# as the rest of this project.
def compute_tillage_yield_mean(combined: gpd.GeoDataFrame) -> float:
    tillage = combined[combined["tillage_class"].notna()]
    return float(tillage["yield_mean"].mean())


def compute_tillage_yield_std(combined: gpd.GeoDataFrame) -> float:
    """Population spread of yield_mean ACROSS tillage parcels (mostly
    reflecting crop-mix variation, e.g. wheat vs barley vs beans, since
    within a single crop type this v1 model has no spatial variation --
    see yield_model_v1.py's documented limitation). Same "spread, not
    per-parcel confidence interval" logic as compute_branch_stds()."""
    tillage = combined[combined["tillage_class"].notna()]
    return float(tillage["yield_mean"].std())


def compute_grassland_index_std(combined: gpd.GeoDataFrame) -> float:
    """Population spread of the grassland productivity index across
    grassland parcels (real region-to-region variation, unlike
    tillage's -- see yield_model_v1.py)."""
    grass = combined[combined["is_grassland"]]
    return float(grass["grassland_index_mean"].std())


def load_area_and_baseline():
    """
    Loads real parcel area (recovering the gap noted above) and each
    parcel's ORIGINAL land_cover_class, keyed by parcel_id (row
    position, matching build_parcel_graph.py's assignment).
    """
    combined = gpd.read_file(COMBINED_PATH).reset_index(drop=True)
    combined["parcel_id"] = combined.index
    combined["area_ha"] = combined.geometry.area / 10_000
    combined["land_cover_class"] = combined.apply(classify_land_cover, axis=1)
    return combined


def filter_reallocatable(parcel_ids: list[int], combined: gpd.GeoDataFrame) -> list[int]:
    """
    Drops any parcel whose ORIGINAL crop is genuinely immutable
    infrastructure (see IMMUTABLE_CLASSES) from the maskable set --
    these should never be offered as reallocation targets, regardless
    of what the generative model might sample for them. They remain
    in the graph as unmasked NEIGHBOR context; they're just never
    themselves a masking target.
    """
    region = combined[combined["parcel_id"].isin(parcel_ids)]
    is_immutable = region["crop"].str.upper().isin(IMMUTABLE_CLASSES)
    n_dropped = is_immutable.sum()
    if n_dropped:
        print(f"Excluded {n_dropped} immutable-infrastructure parcels "
              f"(buildings/roads/quarries/water) from the reallocation "
              "target set -- these remain as fixed context, not candidates.")
    return region.loc[~is_immutable, "parcel_id"].tolist()


def score_scenario(parcel_ids: list[int], new_classes: dict, combined: gpd.GeoDataFrame,
                    branch_carbon_means: dict, branch_carbon_stds: dict,
                    tillage_yield_mean: float, tillage_yield_std: float,
                    grassland_index_std: float) -> dict:
    """
    new_classes: {parcel_id: sampled_class} for the masked region --
    the output of one draw from sample_reallocation().

    Uncertainty propagation: each parcel's NEW value carries the
    branch's population-spread std (see compute_branch_stds's
    docstring for why this, not per-parcel confidence intervals, is
    the right uncertainty here). Parcel outcomes are treated as
    independent draws around their branch mean -- a defensible
    approximation for spatial heterogeneity, NOT the same as
    uncertainty in the branch mean itself (which is small, given the
    large sample sizes most branches have). Aggregate variance for a
    SUM of independent terms is the sum of their variances, so:
        total_std = sqrt( sum_i (area_i * branch_std_i)^2 )
    """
    region = combined[combined["parcel_id"].isin(parcel_ids)].copy()
    region["new_class"] = region["parcel_id"].map(new_classes)

    # --- Carbon: area-weighted mean AND propagated uncertainty ---
    region["new_carbon_per_ha"] = region["new_class"].map(branch_carbon_means)
    region["new_carbon_std_per_ha"] = region["new_class"].map(branch_carbon_stds)
    total_carbon_t = (region["area_ha"] * region["new_carbon_per_ha"]).sum()
    total_carbon_t_std = np.sqrt(
        ((region["area_ha"] * region["new_carbon_std_per_ha"].fillna(0)) ** 2).sum()
    )

    baseline_carbon_t = (region["area_ha"] * region["carbon_t_c_per_ha_mean"]).sum()
    carbon_delta_t = total_carbon_t - baseline_carbon_t
    # Baseline is treated as known (it's measured, not proposed) --
    # the delta's uncertainty is dominated by the proposed scenario's
    # uncertainty, so carbon_delta shares total_carbon_t_std.

    # --- Area by resulting class ---
    area_by_class = region.groupby("new_class")["area_ha"].sum().to_dict()

    # --- Tillage yield estimate + uncertainty (only meaningful for tillage area) ---
    tillage_area = area_by_class.get("tillage", 0.0)
    tillage_yield_estimate_t = tillage_area * tillage_yield_mean
    tillage_yield_estimate_t_std = tillage_area * tillage_yield_std

    # --- Grassland productivity index (area-weighted average, unitless) ---
    grassland_rows = region[region["new_class"] == "grassland"]
    if len(grassland_rows) > 0 and grassland_rows["area_ha"].sum() > 0:
        grass_index_mean = np.average(
            grassland_rows["grassland_index_mean"].fillna(1.0),
            weights=grassland_rows["area_ha"],
        )
    else:
        grass_index_mean = None

    # --- Disruption: how much of the region actually changed class ---
    # A carbon-gain number alone hides its cost. This reports what
    # fraction of the region's area was reallocated away from its
    # baseline class to achieve that gain -- a cheap, honest proxy for
    # "how disruptive is this scenario," pending a more real cost model
    # (compensation, replanting cost, etc.) later.
    region["changed"] = region["new_class"] != region["land_cover_class"]
    disrupted_area_ha = region.loc[region["changed"], "area_ha"].sum()
    total_area_ha = region["area_ha"].sum()
    disruption_fraction = disrupted_area_ha / total_area_ha if total_area_ha > 0 else 0.0

    return {
        "total_carbon_t": total_carbon_t,
        "total_carbon_t_std": total_carbon_t_std,
        "carbon_delta_vs_baseline_t": carbon_delta_t,
        "area_by_class_ha": area_by_class,
        "tillage_yield_estimate_t": tillage_yield_estimate_t,
        "tillage_yield_estimate_t_std": tillage_yield_estimate_t_std,
        "grassland_productivity_index_mean": grass_index_mean,
        "grassland_productivity_index_std": grassland_index_std if grass_index_mean is not None else None,
        "disrupted_area_ha": disrupted_area_ha,
        "disruption_fraction": disruption_fraction,
    }


def generate_and_score_scenarios(parcel_ids: list[int], n_scenarios: int = 10,
                                  temperature: float = 1.0) -> pd.DataFrame:
    model, data, meta, idx_to_class = load_model_and_graph()
    combined = load_area_and_baseline()
    branch_means = compute_branch_means(COMBINED_PATH)
    branch_stds = compute_branch_stds(COMBINED_PATH)
    tillage_yield_mean = compute_tillage_yield_mean(combined)
    tillage_yield_std = compute_tillage_yield_std(combined)
    grassland_index_std = compute_grassland_index_std(combined)

    parcel_ids = filter_reallocatable(parcel_ids, combined)

    samples = sample_reallocation(parcel_ids, model, data, meta, idx_to_class,
                                   n_samples=n_scenarios, temperature=temperature)

    rows = []
    for sample_id in samples["sample_id"].unique():
        this_sample = samples[samples["sample_id"] == sample_id]
        new_classes = dict(zip(this_sample["parcel_id"], this_sample["sampled_class"]))
        score = score_scenario(parcel_ids, new_classes, combined, branch_means, branch_stds,
                                tillage_yield_mean, tillage_yield_std, grassland_index_std)
        score["sample_id"] = sample_id
        rows.append(score)

    result = pd.DataFrame(rows).sort_values("carbon_delta_vs_baseline_t", ascending=False)
    return result.reset_index(drop=True)


if __name__ == "__main__":
    # Smoke test -- replace with a real region of interest (e.g. all
    # parcel_ids within a chosen county or soil association).
    example_parcel_ids = list(range(100, 150))

    scenarios = generate_and_score_scenarios(example_parcel_ids, n_scenarios=10)

    display = scenarios[[
        "sample_id", "total_carbon_t", "total_carbon_t_std",
        "carbon_delta_vs_baseline_t", "disrupted_area_ha", "disruption_fraction",
        "tillage_yield_estimate_t", "tillage_yield_estimate_t_std",
        "grassland_productivity_index_mean", "grassland_productivity_index_std",
    ]].copy()
    display.columns = [
        "Scenario", "Total C (t)", "Total C ±", "C Gain vs Baseline (t)",
        "Disrupted (ha)", "Disruption %", "Tillage Yield (t)", "Tillage Yield ±",
        "Grassland Index", "Grassland Index ±",
    ]
    display["Disruption %"] = (display["Disruption %"] * 100).round(1)

    print("\n--- Scenario tradeoff table (sorted by carbon gain) ---")
    print(display.round(1).to_string(index=False))

    print("\n--- Area by class per scenario (ha) ---")
    for _, row in scenarios.iterrows():
        area_str = ", ".join(f"{cls}={ha:.1f}" for cls, ha in row["area_by_class_ha"].items())
        print(f"  Scenario {row['sample_id']}: {area_str}")
