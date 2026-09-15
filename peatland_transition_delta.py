"""
Peatland transition-delta v1 -- companion to carbon_transition_delta.py,
scoped specifically to transitions BETWEEN LAND USES ON PEAT (industrial
extraction / forest-on-peat / grassland-on-peat / residual peatland),
matching Habib & Connolly (2023)'s four classes.

WHY THIS IS SEPARATE FROM carbon_transition_delta.py, NOT AN EXTRA BRANCH
IN IT:
    carbon_transition_delta.py's branch means are computed LIVE from
    parcel_yield_carbon.gpkg. That works for tillage/grassland/forestry/
    peatland because each of those is a distinct value in the combined
    data. But every LPIS parcel classified as peat lands in ONE bucket
    there (carbon_source == "peatland_literature") at a single flat
    value (BOG_CARBON_MEAN = 705.0, see carbon_model_v1.py) -- there is
    no sub-class in the parcel data distinguishing forest-on-peat from
    grassland-on-peat from bare industrial cutover. So there is nothing
    to compute a live mean FROM for these four sub-classes; they have
    to come from literature emission factors instead, sourced and
    flagged individually below.

WHY EF (FLUX), NOT STOCK-DIFFERENCE:
    carbon_transition_delta.py's delta is a STOCK difference (t C/ha,
    before vs after, no time dimension -- implicitly "eventually, once
    the new steady state is reached"). The peatland GHG literature
    instead reports ANNUAL EMISSION FACTORS (t CO2-C/ha/yr) for a
    given land use, because peat carbon loss/gain from a use change is
    a slow ongoing process (decades), not an instant stock reset. So
    this module's delta needs a TIME HORIZON as an explicit input --
    there is no "steady state" shortcut available.
    Units: literature EFs below are already t CO2-C/ha/yr (carbon
    MASS, not CO2 mass) -- no 44/12 conversion needed. CONFIRM this
    against each source paper before trusting it; some papers report
    plain CO2 flux instead and DO need the conversion.

STATUS: only the afforestation EF is confirmed from a source figure
pulled directly. The other three are None -- placeholders, not
approximations -- do not treat their absence as "zero effect."

Two uses for this module, kept deliberately separate:
    1. Backtest reference: compare its output against observed
       Habib & Connolly transition areas (see backtest_transition_deltas.py)
       -- this is validation, not deployment.
    2. Eventual optimizer input, IF AND ONLY IF the backtest in (1)
       shows the predicted deltas land in a defensible range. Do not
       wire this into the scenario optimizer before that check passes
       -- an unvalidated EF plugged into the Pareto search is worse
       than no peat-transition delta at all, since it would look
       precise while being unchecked.
"""

import pandas as pd


# Reference emission factors, t CO2-C per ha per year.
# Sign convention: NEGATIVE = net source (peat losing carbon, e.g.
# drainage/disturbance), POSITIVE = net sink (peat gaining carbon).
# CONFIRM each sign against its source paper -- do not assume.
PEAT_SUBCLASS_EF_T_CO2C_HA_YR = {
    # Jovani-Sancho, Cummins & Byrne (2021), Ireland-specific Tier 2
    # (replaces IPCC Tier 1 "Forest Land, drained, organic soil"):
    # 153,831 t CO2-C / 260,730 ha stocked afforested peatland.
    # Reported as an EMISSION (net carbon loss) -> negative.
    "forest_on_peat": -153_831 / 260_730,  # -0.590, CONFIRMED

    # Wilson et al. (2015), Ireland/UK-specific Tier 2, Biogeosciences
    # 12:5291. Industrial sites: 1.70 (+/- 0.47) t CO2-C/ha/yr, reported
    # as an EMISSION -> negative. Notably LOWER than IPCC Tier 1 default
    # of 2.8 (+/- 1.7) for this category -- use this, not Tier 1.
    "industrial": -1.70,  # CONFIRMED, +/- 0.47

    # Aitova et al. (2023), Ireland-specific Tier 2 (nutrient-rich
    # grassland on peat): 5.08 t CO2-C/ha/yr, reported as an EMISSION
    # -> negative. IPCC Tier 1 default for the same category is 6.1 --
    # use the Irish Tier 2 figure, not Tier 1, consistent with the
    # industrial case above.
    "grassland_on_peat": -5.08,  # CONFIRMED, Aitova et al. 2023 Tier 2

    # Residual peatland is explicitly heterogeneous in Habib & Connolly's
    # own Table 1 (revegetated bog, remnant high bog, AND domestic
    # cutover mixed together) -- NOT simply "undisturbed." A single EF
    # (e.g. treating it as a near-zero "near-natural" reference state,
    # as IPCC Tier 1 tables often do for undrained peat) would be a
    # real simplification, not a sourced fact -- decide and document
    # this explicitly rather than defaulting silently. PLACEHOLDER.
    "residual_peatland": 0.0,  # ASSUMPTION -- near-natural/undrained reference state, not measured
}


def get_peat_subclass_ef(subclass: str) -> float:
    ef = PEAT_SUBCLASS_EF_T_CO2C_HA_YR.get(subclass)
    if ef is None:
        raise ValueError(
            f"No confirmed EF for peat sub-class '{subclass}' yet -- "
            "source it from the literature before using, don't default "
            "to 0 or an IPCC Tier 1 substitute without flagging that "
            "substitution explicitly."
        )
    return ef


def transition_delta_peat(source_subclass: str, target_subclass: str,
                          years: float, parcel_row=None) -> float:
    """
    Estimated t C/ha CHANGE from converting a parcel from source_subclass
    to target_subclass ON PEAT, over `years`.

    Unlike carbon_transition_delta.py's transition_delta(), this is NOT
    a single before/after stock difference -- it's
    (target_EF - source_EF) * years, i.e. the cumulative divergence in
    annual flux between the two land uses over the given horizon. This
    is what the literature's EFs actually measure; forcing it into a
    single stock-difference number the way the mineral-soil version
    works would misrepresent what's being computed.

    parcel_row: unused in this v1 -- kept for signature parity with
    carbon_transition_delta.py's transition_delta(), so a future
    version keyed on e.g. peat depth or drainage status can use it
    without changing calling code.
    """
    target_ef = get_peat_subclass_ef(target_subclass)
    source_ef = get_peat_subclass_ef(source_subclass)
    return (target_ef - source_ef) * years


if __name__ == "__main__":
    print("Confirmed peat sub-class EFs (t CO2-C/ha/yr):")
    for cls, ef in PEAT_SUBCLASS_EF_T_CO2C_HA_YR.items():
        status = f"{ef:+.3f}" if ef is not None else "NOT YET SOURCED"
        print(f"  {cls}: {status}")

    print("\n--- Example: grassland-on-peat -> forest-on-peat, 14 yr ---")
    try:
        delta = transition_delta_peat("grassland_on_peat", "forest_on_peat", 14)
        print(f"  {delta:+.2f} t C/ha")
    except ValueError as e:
        print(f"  Cannot compute yet: {e}")
