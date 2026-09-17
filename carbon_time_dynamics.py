"""
Carbon time-dynamics v1 -- replaces the instant-swap assumption
(baseline stock -> target branch mean, in year 0) with a trajectory
over time, so scenario outputs can report "carbon sequestered by
[policy year]" rather than an idealized steady-state jump.

WHAT'S REAL VS. ASSUMED -- read before trusting these numbers:

  AFFORESTATION (-> forestry): the ONLY transition with a real cited
  anchor -- peak sequestration ~13 t CO2/ha/yr around year 15
  (converted to ~3.54 t C/ha/yr; CO2->C via the standard 3.67 factor).
  Modeled as a Gamma-distribution-shaped annual rate curve:
    - PEAK TIMING (mode = 15 years): REAL, cited.
    - SHAPE PARAMETER (k=3, controls how "peaked" vs "spread out" the
      curve is): ASSUMED. Only one real data point exists (peak
      location + magnitude), which isn't enough to fully determine a
      curve shape -- k=3 is a reasonable, moderately-skewed default,
      not independently sourced. Flagged, not hidden.
    - TOTAL EVENTUAL GAIN: anchored to this project's OWN branch-mean
      carbon figures (target - baseline), not the citation -- these
      are numbers this project has already built and trusted
      elsewhere, so the trajectory's endpoint matches the rest of the
      model rather than introducing a second, inconsistent ground
      truth.
    - CROSS-CHECK: given the above, the curve's IMPLIED peak rate is
      computed and printed -- compare it to the cited ~13 t CO2/ha/yr
      as an informal validation, not a hard constraint that was forced
      to match.

  ALL OTHER TRANSITIONS (grassland<->tillage, ->peatland, etc.):
  NO equivalent literature curve was found for these in this project.
  Modeled with a simple exponential approach-to-target using an
  ASSUMED, UNSOURCED default characteristic timescale (see
  DEFAULT_TIMESCALE_YEARS below). This is a placeholder, explicitly
  weaker than the afforestation curve -- do not present these
  trajectories with the same confidence in a paper without either
  finding a real source or clearly caveating them as illustrative.
"""

import numpy as np
from scipy.stats import gamma as gamma_dist

# --- Afforestation curve parameters ---
AFFORESTATION_PEAK_YEAR = 15          # REAL, cited
AFFORESTATION_PEAK_RATE_T_CO2 = 13.0  # REAL, cited
CO2_TO_C_FACTOR = 3.67                # standard molecular-weight ratio, not a modeling choice
AFFORESTATION_PEAK_RATE_T_C = AFFORESTATION_PEAK_RATE_T_CO2 / CO2_TO_C_FACTOR

AFFORESTATION_SHAPE_K = 3.0  # ASSUMED default -- see module docstring

# --- Default timescale for all other (unsourced) transitions ---
DEFAULT_TIMESCALE_YEARS = 20.0  # ASSUMED, UNSOURCED placeholder -- see docstring


def _gamma_scale_for_mode(mode: float, shape_k: float) -> float:
    """theta such that the Gamma(shape_k, theta) PDF's mode equals `mode`.
    Mode of Gamma(k, theta) for k>1 is theta*(k-1)."""
    return mode / (shape_k - 1)


def afforestation_trajectory(baseline_stock: float, target_stock: float, years: np.ndarray) -> dict:
    """
    Models the carbon stock trajectory for a transition TO forestry.

    total_gain (target_stock - baseline_stock) is anchored to this
    project's own branch-mean figures -- see module docstring for why
    this, not the literature's peak-rate figure, sets the curve's
    total scale.

    Returns dict with 'stock_at_year' (cumulative stock, baseline ->
    baseline+total_gain), 'annual_rate_t_c' (rate curve), and
    'implied_peak_rate_t_co2' (cross-check against the citation).
    """
    total_gain = target_stock - baseline_stock
    theta = _gamma_scale_for_mode(AFFORESTATION_PEAK_YEAR, AFFORESTATION_SHAPE_K)

    pdf_values = gamma_dist.pdf(years, a=AFFORESTATION_SHAPE_K, scale=theta)
    cdf_values = gamma_dist.cdf(years, a=AFFORESTATION_SHAPE_K, scale=theta)

    annual_rate_t_c = total_gain * pdf_values
    stock_at_year = baseline_stock + total_gain * cdf_values

    peak_pdf = gamma_dist.pdf(AFFORESTATION_PEAK_YEAR, a=AFFORESTATION_SHAPE_K, scale=theta)
    implied_peak_rate_t_c = total_gain * peak_pdf
    implied_peak_rate_t_co2 = implied_peak_rate_t_c * CO2_TO_C_FACTOR

    return {
        "stock_at_year": stock_at_year,
        "annual_rate_t_c": annual_rate_t_c,
        "implied_peak_rate_t_co2": implied_peak_rate_t_co2,
        "cited_peak_rate_t_co2": AFFORESTATION_PEAK_RATE_T_CO2,
    }


def generic_trajectory(baseline_stock: float, target_stock: float, years: np.ndarray,
                        timescale: float = DEFAULT_TIMESCALE_YEARS) -> dict:
    """
    Simple exponential approach-to-target for transitions with no
    known literature curve. UNSOURCED default timescale -- see module
    docstring. stock(t) = target - (target-baseline)*exp(-t/timescale).
    """
    total_gain = target_stock - baseline_stock
    stock_at_year = target_stock - total_gain * np.exp(-years / timescale)
    annual_rate_t_c = (total_gain / timescale) * np.exp(-years / timescale)
    return {"stock_at_year": stock_at_year, "annual_rate_t_c": annual_rate_t_c}


def carbon_at_year(source_class: str, target_class: str, baseline_stock: float,
                    target_stock: float, years_elapsed: float) -> float:
    """
    Main entry point: carbon stock (t C/ha) at a given number of years
    after a proposed reallocation, replacing the old instant-swap
    assumption. Dispatches to the afforestation curve (real citation)
    for transitions TO forestry, and the generic placeholder curve
    otherwise.
    """
    years = np.array([years_elapsed])
    if target_class == "forestry" and source_class != "forestry":
        result = afforestation_trajectory(baseline_stock, target_stock, years)
    else:
        result = generic_trajectory(baseline_stock, target_stock, years)
    return float(result["stock_at_year"][0])


if __name__ == "__main__":
    # Sanity check using this project's real grassland->forestry branch means.
    baseline, target = 331.4, 448.1
    years = np.arange(0, 61, 5)

    result = afforestation_trajectory(baseline, target, years)
    print("Grassland -> Forestry trajectory (real branch-mean anchors):")
    print(f"{'Year':>5} {'Stock (t C/ha)':>16} {'Rate (t C/ha/yr)':>18}")
    for y, s, r in zip(years, result["stock_at_year"], result["annual_rate_t_c"]):
        print(f"{y:>5} {s:>16.1f} {r:>18.3f}")

    print(f"\nModel's implied peak rate: {result['implied_peak_rate_t_co2']:.1f} t CO2/ha/yr")
    print(f"Cited literature peak rate: {result['cited_peak_rate_t_co2']:.1f} t CO2/ha/yr")
    print(f"(These are independent checks, not forced to match -- see module docstring)")
