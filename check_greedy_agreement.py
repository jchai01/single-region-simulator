"""
Diagnostic: is "greedy" well-defined on the real region?

Baseline carbon values come from soil-association polygons, so many parcels have EXACTLY the
same gain per hectare. A density-greedy then has to order tied parcels somehow, and when the
budget boundary falls inside a group of ties the result depends on that arbitrary order: two
correct implementations gave results differing by up to ~0.6% on this region.

This script reports, on the real region:
  1. whether baseline_comparison.greedy_baseline and mckp_tools.solve_greedy now select the
     SAME parcels (they use one tie-break rule, parcel_id ascending);
  2. how many positive-gain parcels share their gain per hectare with another parcel;
  3. how much greedy's result moves with the tie order (ascending/descending parcel_id, larger/
     smaller area first, and 30 random orders), for both greedy rules. This is the noise
     floor of any statement about greedy at a fraction-of-a-percent level.

Usage (repo root):  python check_greedy_agreement.py
"""
import numpy as np
import pandas as pd

import mckp_tools as mt

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def agreement(problem, solver, inst, budgets=BUDGETS):
    """Same parcels, same carbon, for BOTH greedy rules (stop, and skip-and-fill)."""
    import baseline_comparison as bc
    ids = list(problem.parcel_ids)
    orig = dict(zip(problem.combined["parcel_id"], problem.combined["land_cover_class"]))
    rows = []
    for d in budgets:
        row = {"budget": d}
        for tag, fill in (("stop", False), ("fill", True)):
            res, new = bc.greedy_baseline(ids, problem.combined, problem.branch_means, problem.branch_stds, d, fill=fill)
            a = {p for p in ids if new[p] != orig[p]}
            chosen, g = mt.solve_greedy(inst, d, fill=fill)
            b = {ids[inst.group[m]] for m in chosen}
            row[f"{tag}: baseline_comparison"] = res["carbon_delta_vs_baseline_t"]
            row[f"{tag}: mckp_tools"] = g
            row[f"{tag}: parcels_that_differ"] = len(a ^ b)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["parcels_that_differ"] = out["stop: parcels_that_differ"] + out["fill: parcels_that_differ"]
    return out


def tie_prevalence(inst):
    ok = np.flatnonzero((inst.gain > 0) & (inst.weight > 0))
    dens = np.round(inst.gain[ok] / inst.weight[ok], 9)
    grp = inst.group[ok]
    best = pd.DataFrame({"g": grp, "d": dens}).sort_values(["g", "d"], ascending=[True, False]).drop_duplicates("g")
    counts = best["d"].value_counts()
    tied = int((best["d"].map(counts) > 1).sum())
    return {"positive_gain_parcels": len(best), "distinct_densities": int(counts.size),
            "parcels_sharing_density": tied, "largest_tie_group": int(counts.max())}


def tie_sensitivity(solver, inst, budgets=BUDGETS, n_random=30, seed=0):
    rng = np.random.default_rng(seed)
    n = inst.n_groups
    area = solver.area
    orders = {"parcel_id asc (default)": np.arange(n), "parcel_id desc": -np.arange(n),
              "larger area first": -area, "smaller area first": area}
    rows = []
    for d in budgets:
        for fill in (False, True):
            vals = {k: mt.solve_greedy(inst, d, fill=fill, tie_priority=v)[1] for k, v in orders.items()}
            rnd = [mt.solve_greedy(inst, d, fill=fill, tie_priority=rng.permutation(n))[1] for _ in range(n_random)]
            allv = list(vals.values()) + rnd
            rows.append({"budget": d, "rule": "fill" if fill else "stop", "default": vals["parcel_id asc (default)"],
                         "min": min(allv), "max": max(allv), "range_%_of_max": 100 * (max(allv) - min(allv)) / max(allv)})
    return pd.DataFrame(rows)


def main():
    import solve_exact_front as sef
    from nsga2_scenario_search import evaluate_batch

    problem = sef.build_problem()
    solver = sef.ExactSolver(problem, evaluate_batch, True)
    inst = mt.parcel_instance(solver)
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    print("\n=== 1. Do the two greedy implementations select the same parcels? ===")
    ag = agreement(problem, solver, inst)
    print(ag.to_string(index=False))
    print("-> " + ("They agree exactly, for both the stop and the fill rule." if (ag["parcels_that_differ"] == 0).all()
                    else "They still differ. Send me this table: the differing parcels need inspecting."))

    t = tie_prevalence(inst)
    print("\n=== 2. How many parcels tie? ===")
    print(f"{t['parcels_sharing_density']} of {t['positive_gain_parcels']} positive-gain parcels share their gain per hectare with "
          f"at least one other parcel ({t['distinct_densities']} distinct values; largest tie group: {t['largest_tie_group']} parcels).")

    print("\n=== 3. How much does greedy move with the tie order? (carbon, t C) ===")
    ts = tie_sensitivity(solver, inst)
    print(ts.to_string(index=False))
    worst = ts.groupby("rule")["range_%_of_max"].max()
    print(f"\nLargest spread over tie orders: stop-greedy {worst['stop']:.2f}%, fill-greedy {worst['fill']:.2f}%. "
          "Claims about greedy at a finer resolution than this are within tie-order noise; report the rule and its tie-break.")


if __name__ == "__main__":
    main()
