"""
Carbon transition-delta v1 -- estimates how a parcel's carbon stock
would change under a proposed land-use reallocation.

DESIGN: branch-mean delta (simple, swappable)
    delta = mean_carbon[target_class] - mean_carbon[source_class]

    Branch means are computed LIVE from the actual combined parcel
    data (parcel_yield_carbon.gpkg), not hardcoded -- so this stays
    correct as the underlying models are refined, rather than baking
    in stale numbers from one point in time.

KNOWN LIMITATION IN transition_delta() (Option A, below) -- still true,
kept for the cases where it's still the right tool (see hybrid below):
    This delta reflects a GEOGRAPHIC pattern, not a controlled land-use
    effect. The carbon estimate itself is computed purely from soil
    association (a spatial join, independent of tillage/grassland
    classification) -- so the observed tillage-vs-grassland difference
    reflects WHERE each land use happens to be concentrated in Ireland,
    not a measured or modeled effect of converting one specific parcel
    from one use to the other. Applying this delta to an individual
    parcel assumes it would shift toward the NATIONAL AVERAGE of its
    target class, which may not hold for that parcel's actual soil.

HYBRID UPGRADE -- NOW IMPLEMENTED (see hybrid_transition_target_stock()
below): SOLUM's land-use coefficients, which DO control for soil
cluster, applied as a RELATIVE multiplier anchored to this dataset's
own absolute stock levels rather than SOLUM's own intercept. This
corrects the geographic-confounding limitation above -- but ONLY for
transitions between mineral-soil classes (tillage/grassland/forestry),
since SOLUM's model explicitly excludes organic/peat soils. Any
transition involving peatland still uses Option A (transition_delta())
as a deliberate fallback, not a missing case -- see
hybrid_transition_target_stock()'s docstring for why.

Usage:
    means = compute_branch_means("data/parcel_yield_carbon.gpkg")
    delta = transition_delta("tillage", "grassland", means)
    # delta = expected t C/ha CHANGE if a tillage parcel converts to
    # grassland, per this v1's national-average-branch assumption
    # (Option A -- geographically confounded, see limitation above).

    target_stock = hybrid_transition_target_stock(
        "tillage", "grassland", baseline_stock=this_parcels_own_value,
        branch_means=means,
    )
    # target_stock = this SPECIFIC parcel's expected eventual t C/ha
    # under the transition, using SOLUM's real soil-controlled relative
    # effect anchored to its own measured baseline (mineral-soil
    # transitions only; falls back to Option A's branch mean for any
    # transition involving peatland).
"""

import math

import geopandas as gpd
import pandas as pd


def classify_land_cover(row) -> str:
    """
    Assigns a parcel to one of six land-cover classes for the purpose
    of transition-delta lookups. Distinct from carbon_model_v1.py's
    'carbon_source' column, which groups by DATA SOURCE (sis_measured_
    series covers both tillage AND grassland) rather than by land-use
    class -- this function separates tillage from grassland, which the
    delta calculation needs.

    IMPORTANT: 'excluded' means truly non-agricultural, zero-carbon
    land (buildings, roads, scheme land) -- checked explicitly via
    is_excluded, NOT used as a catch-all default. A naive catch-all
    would incorrectly lump in the yield model's ~15,000 'unresolved'
    parcels (real crops like maize/potatoes with no yield source, but
    which still have a real, nonzero measured soil carbon value from
    the spatial SIS join) -- confirmed this was inflating the
    'excluded' branch mean from the true 0.0 up to ~9.7 t C/ha.
    Those unresolved-but-real crops get their own 'unresolved_crop'
    bucket instead, so 'excluded' stays a clean zero reference.
    """
    if pd.notna(row.get("tillage_class")):
        return "tillage"
    if row.get("is_grassland"):
        return "grassland"
    if row.get("carbon_source") == "nfi_blended_average":
        return "forestry"
    if row.get("carbon_source") == "peatland_literature":
        return "peatland"
    if row.get("is_excluded"):
        return "excluded"
    return "unresolved_crop"


def compute_branch_means(combined_path: str = "data/parcel_yield_carbon.gpkg") -> dict:
    """
    Computes live mean carbon stock (t C/ha) per land-cover class from
    the actual combined parcel data -- NOT hardcoded, so this reflects
    whatever the current carbon model actually says, and stays correct
    as that model is refined.
    """
    gdf = gpd.read_file(combined_path)
    gdf["land_cover_class"] = gdf.apply(classify_land_cover, axis=1)

    means = gdf.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].mean().to_dict()
    counts = gdf.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].count().to_dict()

    print("Branch mean carbon stock (t C/ha), computed live from combined data:")
    for cls in means:
        print(f"  {cls}: {means[cls]:.1f} (n={counts.get(cls, 0)})")

    return means


def compute_branch_stds(combined_path: str = "data/parcel_yield_carbon.gpkg") -> dict:
    """
    Computes the POPULATION spread of carbon_t_c_per_ha_mean within
    each land-cover class -- i.e. how much real carbon density varies
    ACROSS the parcels already in that branch (soil-association-level
    heterogeneity), not the per-parcel measurement confidence interval.

    This is the deliberately correct uncertainty to use when scoring a
    proposed reallocation: when a parcel is reassigned to a new class,
    we genuinely don't know where in that class's real distribution the
    reallocated parcel would land -- that's captured by the branch's
    spread, not by the confidence interval on any single existing
    parcel's point estimate (carbon_t_c_per_ha_std), which reflects a
    different, narrower question ("how sure are we about THIS parcel's
    current value").
    """
    gdf = gpd.read_file(combined_path)
    gdf["land_cover_class"] = gdf.apply(classify_land_cover, axis=1)
    stds = gdf.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].std().to_dict()
    return stds


def transition_delta(source_class: str, target_class: str, branch_means: dict,
                      parcel_row=None) -> float:
    """
    Estimated t C/ha CHANGE from converting a parcel from source_class
    to target_class.

    parcel_row: unused in this v1 (branch-mean) version -- accepted
    here so a future soil-cluster-aware version can use the parcel's
    own soil_cluster/Associat_S without changing how callers invoke
    this function.

    Returns NaN (with a warning) if either class isn't in branch_means
    -- e.g. if the combined dataset had zero parcels of that class, or
    a typo in the class name.
    """
    if source_class not in branch_means or target_class not in branch_means:
        print(f"Warning: '{source_class}' or '{target_class}' not found in "
              f"branch_means (available: {list(branch_means.keys())}). "
              "Returning NaN.")
        return float("nan")

    return branch_means[target_class] - branch_means[source_class]


# SOLUM's land-use (Flu) coefficients (Saunders et al. 2022, EPA Report
# No. 422, Table 3.2), mapped to this project's class names. Peatland
# is deliberately absent -- SOLUM's GLM explicitly excludes organic
# soils, so it has no coefficient to offer for that class at all.
SOLUM_FLU_COEF = {
    "tillage": 0.0,       # SOLUM's "cropland", reference level
    "grassland": 0.2673,  # SOLUM's "grassland"; management-regime term
                          # (FMG: improved/unimproved/transitioning) is
                          # NOT modeled here -- this project's class
                          # taxonomy doesn't track management regime,
                          # so FMG is implicitly treated as a wash (0)
                          # for all grassland. Stated simplification.
    "forestry": 0.36125,  # SOLUM's "forest"
}


def hybrid_transition_target_stock(source_class: str, target_class: str,
                                   baseline_stock: float, branch_means: dict) -> float:
    """
    Returns the EVENTUAL (steady-state) target stock a SPECIFIC parcel
    should trend toward under a transition -- the value to pass as
    carbon_time_dynamics.carbon_at_year()'s `target_stock` argument,
    replacing a flat branch_means[target_class] lookup.

    THE HYBRID: for transitions between two MINERAL-SOIL classes
    (tillage/grassland/forestry, i.e. both classes present in
    SOLUM_FLU_COEF), scales THIS PARCEL'S OWN measured baseline_stock
    by SOLUM's land-use coefficients, rather than jumping to the
    target class's population branch mean (Option A's geographic
    confound -- see module docstring).

    WHY THIS IS VALID WITHOUT KNOWING THIS PARCEL'S SOIL CLUSTER:
    SOLUM's full model is SOC = exp(a + CL + Flu + FMG). The intercept
    (a) and soil-cluster term (CL) depend on the parcel's SOIL, not its
    land use -- they are IDENTICAL whether evaluating the source or
    target land use for the SAME parcel. Taking the ratio of SOLUM's
    predicted SOC between target and source land use therefore cancels
    BOTH a and CL exactly:
        SOC_target / SOC_source = exp(Flu_target - Flu_source)
    independent of soil cluster (verified numerically: computing this
    ratio under two different SOLUM soil-cluster coefficients gives the
    identical result to 9 decimal places). This is what lets this
    function borrow SOLUM's real, soil-controlled RELATIVE effect and
    apply it to this project's own absolute measurements, without
    needing a per-parcel soil-cluster lookup at this call site.

    FALLBACK, DELIBERATE NOT MISSING: if either class is absent from
    SOLUM_FLU_COEF (i.e. the transition involves peatland, in either
    direction), returns branch_means[target_class] unchanged (Option
    A) instead. SOLUM's GLM explicitly excludes organic/peat soils --
    applying its mineral-soil coefficients to a peat transition would
    be a scientific misapplication, not an improvement.
    """
    if source_class in SOLUM_FLU_COEF and target_class in SOLUM_FLU_COEF:
        relative_multiplier = math.exp(
            SOLUM_FLU_COEF[target_class] - SOLUM_FLU_COEF[source_class]
        )
        return baseline_stock * relative_multiplier
    return branch_means[target_class]


if __name__ == "__main__":
    means = compute_branch_means()

    print("\n--- Option A (branch-mean, geographically confounded) ---")
    for source, target in [("tillage", "grassland"), ("grassland", "tillage"),
                            ("tillage", "forestry"), ("grassland", "forestry"),
                            ("tillage", "peatland")]:
        delta = transition_delta(source, target, means)
        direction = "gain" if delta > 0 else "loss"
        print(f"  {source} -> {target}: {delta:+.1f} t C/ha ({direction})")

    print("\n--- Hybrid (SOLUM-shaped, anchored to a SPECIFIC parcel's own baseline) ---")
    print("Example: two DIFFERENT tillage parcels with different own baselines,")
    print("same transition -- hybrid gives each its OWN answer; Option A would")
    print("give both the SAME answer (the population branch mean), which is")
    print("exactly the confound being corrected.")
    for own_baseline in [140.0, 220.0]:
        target_stock = hybrid_transition_target_stock("tillage", "grassland", own_baseline, means)
        print(f"  tillage (own baseline {own_baseline:.1f}) -> grassland: "
              f"target_stock={target_stock:.1f} (delta={target_stock - own_baseline:+.1f} t C/ha)")

    print("\n  tillage -> peatland (falls back to Option A, SOLUM has no organic-soil coefficient):")
    fallback_target = hybrid_transition_target_stock("tillage", "peatland", 180.0, means)
    print(f"  target_stock={fallback_target:.1f} (== branch_means['peatland']={means['peatland']:.1f})")
