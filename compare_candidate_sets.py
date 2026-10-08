"""
Experiment 3 -- does treating the 14 maize / fodder beet / kale parcels as tillage
candidates change anything?

'unresolved_crop' parcels are real cropland that the yield model had no source for, so
they are currently excluded from the candidate set. The ones whose crop is a plain
arable/forage crop (generate_scenarios.TILLAGE_LIKE_CROPS) are tillage in every sense
the carbon transition cares about. This script compares the candidate set as it is now
with the set that includes them as tillage, on the exact optimum and on greedy, and
reports whether the optimum actually converts any of them (at the same budget fraction, and at
the same hectares so the value of the extra options is not confused with a larger budget).

The change has since been ADOPTED (generate_scenarios.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED = True)
because this script found it worth +5.5-6.1% of the optimum at budgets >= 0.3 (same hectares),
so this script now stands as the justification and as a way to re-measure the effect. It does
not rerun NSGA-II. The GNN does not need retraining: the reclassification happens after the
graph and class vocabulary are fixed.

Usage (repo root):  python compare_candidate_sets.py
"""
import numpy as np
import pandas as pd

from mckp_tools import parcel_instance, solve_exact, solve_greedy

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def run(problem_a, problem_b, evaluate_batch, budgets=BUDGETS):
    """Two comparisons per budget:
      * same budget FRACTION (what rerunning the pipeline would show) -- but the larger region
        also has more hectares to change, which mixes the new options with a bigger budget;
      * same HECTARES (the budget is rescaled by the region sizes), which isolates the pure
        value of having the extra candidates. The verdict is based on this one."""
    import solve_exact_front as sef
    sa = sef.ExactSolver(problem_a, evaluate_batch, True)
    sb = sef.ExactSolver(problem_b, evaluate_batch, True)
    ia, ib = parcel_instance(sa), parcel_instance(sb)
    new_ids = set(problem_b.parcel_ids) - set(problem_a.parcel_ids)
    pids_b = np.asarray(problem_b.parcel_ids)
    scale = sa.total_area / sb.total_area
    rows = []
    for d in budgets:
        _, opt_a = solve_exact(ia, d)
        _, opt_b = solve_exact(ib, d)
        cb, opt_bh = solve_exact(ib, d * scale)                     # same hectares as the current region
        _, g_a = solve_greedy(ia, d)
        _, g_b = solve_greedy(ib, d)
        _, g_bh = solve_greedy(ib, d * scale)
        conv = [m for m in cb if pids_b[ib.group[m]] in new_ids]
        rows.append({"budget": d, "optimum_now": opt_a,
                     "optimum_with_added": opt_b, "change_%": 100 * (opt_b / opt_a - 1),
                     "optimum_with_added_same_ha": opt_bh, "option_value_%": 100 * (opt_bh / opt_a - 1),
                     "greedy_now": g_a, "greedy_with_added": g_b, "greedy_with_added_same_ha": g_bh,
                     "added_parcels_converted": len(conv),
                     "carbon_from_them_t": float(ib.gain[conv].sum()) if conv else 0.0})
    return pd.DataFrame(rows), new_ids, sb, ib


def describe_new(problem_b, solver_b, inst_b, new_ids):
    pids = np.asarray(problem_b.parcel_ids)
    idx = [i for i, p in enumerate(pids) if p in new_ids]
    if not idx:
        return "No parcels were added."
    best = np.zeros(solver_b.n)
    np.maximum.at(best, inst_b.group, inst_b.gain)
    base = problem_b.combined.set_index("parcel_id").loc[pids[idx], "carbon_t_c_per_ha_mean"]
    return (f"{len(idx)} parcels added ({solver_b.area[idx].sum():,.1f} ha, "
            f"{100 * solver_b.area[idx].sum() / solver_b.total_area:.1f}% of the new region); "
            f"mean baseline {np.average(base, weights=solver_b.area[idx]):.1f} t C/ha; "
            f"best-case carbon if all converted to their best class: {best[idx].sum():,.0f} t C.")


def main():
    import generate_scenarios as gs
    import solve_exact_front as sef
    from nsga2_scenario_search import evaluate_batch

    combined = gs.load_area_and_baseline(reclassify_tillage_like=False)
    combined_b = gs.reclassify_unresolved_tillage_like(combined.copy())
    print("\n--- candidate set as it is now ---")
    pa = sef.build_problem(combined=combined)
    print("\n--- candidate set with maize/fodder beet/kale as tillage ---")
    pb = sef.build_problem(combined=combined_b)

    table, new_ids, sb, ib = run(pa, pb, evaluate_batch)
    pd.set_option("display.width", 240); pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"\n{len(pa.parcel_ids)} candidates now -> {len(pb.parcel_ids)} with the reclassification.")
    print(describe_new(pb, sb, ib, new_ids))
    print("\n=== Effect on the exact optimum and on greedy ===")
    print(table.to_string(index=False))
    big = table["option_value_%"].abs().max()
    print("\n'change_%' compares at the same budget FRACTION (the region also grows); 'option_value_%' compares at the same "
          "HECTARES, isolating what the extra candidates are worth on their own.")
    print(f"Largest option value at any budget: {big:.2f}%. "
          + ("Negligible -- keep the current, more conservative candidate set (or adopt it; nothing material depends on it)."
             if big < 1.0 else "Material -- consider adopting it and rerunning the baselines and NSGA-II."))
    table.to_csv("data/candidate_set_comparison.csv", index=False)
    print("Wrote data/candidate_set_comparison.csv")


if __name__ == "__main__":
    main()
