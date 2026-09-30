"""
Backtest: does the carbon model's predicted transition delta land in the
right ballpark against real observed transitions?

This is a DIFFERENT validation from the spatial-agreement check already
done -- that validated the BASELINE BOG classification. This validates
the TRANSITION-DELTA logic, which currently doesn't exist for peatland
in carbon_model_v1.py (solum_soc() is scoped to mineral soils only --
see its docstring / SOLUM_LANDUSE_COEF, which has no peat/organic path).

Approach: annual emission-factor (EF) based, matching how the peatland
GHG literature actually reports these numbers (t CO2-C/ha/yr), NOT
SOLUM's stock-difference GLM approach, because the two aren't comparable.

    predicted_tC_change = area_transitioned_ha * EF_t_CO2C_per_ha_yr
                           * years_in_period * (12/44 to convert CO2 mass to C, IF your EF is in CO2 not CO2-C -- check units)

Data:
  - Observed transitioned area: Habib & Connolly (2023) Table 5
    (2005->2019 transition matrix, values in ha)
  - Reference EFs: Ireland/UK-specific literature values (Tier 2 where
    available, preferred over IPCC Tier 1 defaults per Wilson et al. 2015
    finding that Tier 1 overestimates for Irish peat)

STATUS: two of three EFs below are placeholders needing the actual
published figure -- flagged individually. Do not run this as-is.
"""

from peatland_transition_delta import PEAT_SUBCLASS_EF_T_CO2C_HA_YR

YEARS_2005_2019 = 14

# Observed transition areas, ha, from Habib & Connolly (2023) Table 5
# (2005 -> 2019 land use change matrix)
OBSERVED_TRANSITIONS_HA = {
    "grassland_to_forest": 34_700,
    "forest_to_grassland": 65_700,
    "industrial_to_forest": 5_100,
    "industrial_to_grassland": 2_800,
    "industrial_to_residual": 17_500,
}

# All four EFs now sourced -- see peatland_transition_delta.py for
# provenance and caveats (residual_peatland is a documented ASSUMPTION,
# the other three are measured Ireland-specific Tier 2 values).
REFERENCE_EF_T_CO2C_HA_YR = PEAT_SUBCLASS_EF_T_CO2C_HA_YR


def backtest_transition(transition_key, source_class, target_class):
    area_ha = OBSERVED_TRANSITIONS_HA[transition_key]
    source_ef = REFERENCE_EF_T_CO2C_HA_YR[source_class]
    target_ef = REFERENCE_EF_T_CO2C_HA_YR[target_class]

    # (target_EF - source_EF) * years -- cumulative divergence in flux
    # between the two land uses, matching transition_delta_peat()'s
    # convention (not a stock difference).
    predicted_tC_per_ha = (target_ef - source_ef) * YEARS_2005_2019
    predicted_total_tC = predicted_tC_per_ha * area_ha

    note = " [residual_peatland side is an ASSUMPTION, not measured]" \
        if "residual" in (source_class, target_class) else ""

    print(f"[{transition_key}] {area_ha:,} ha, {source_class}({source_ef:+.2f}) "
          f"-> {target_class}({target_ef:+.2f}) over {YEARS_2005_2019} yr "
          f"-> {predicted_tC_per_ha:+.2f} t C/ha -> {predicted_total_tC:+,.0f} t C total{note}")
    return predicted_total_tC


if __name__ == "__main__":
    print("=== Transition backtest against Habib & Connolly (2023) Table 5 ===\n")
    backtest_transition("grassland_to_forest", "grassland_on_peat", "forest_on_peat")
    backtest_transition("forest_to_grassland", "forest_on_peat", "grassland_on_peat")
    backtest_transition("industrial_to_forest", "industrial", "forest_on_peat")
    backtest_transition("industrial_to_grassland", "industrial", "grassland_on_peat")
    backtest_transition("industrial_to_residual", "industrial", "residual_peatland")

    print("\nNOTE: these are the carbon model's PREDICTED deltas using literature\n"
          "EFs, applied to Habib & Connolly's OBSERVED transition areas. The\n"
          "actual backtest is comparing these predicted totals against an\n"
          "independently-derived expected range for the same transitions --\n"
          "e.g. the papers' own national emissions estimates (Table 5 of the\n"
          "ScienceDirect peat-soil paper gives ~7.6-8.1 Mt CO2eq/yr nationally\n"
          "for grassland-on-peat alone) -- not just reported as standalone\n"
          "numbers. A result within roughly a factor of 2 of an independent\n"
          "estimate is consistent with the normal Tier 1-vs-Tier 2 spread seen\n"
          "throughout this literature; treat larger gaps as worth investigating,\n"
          "not automatically a modeling failure given how much site-to-site\n"
          "variability this literature itself reports.")
