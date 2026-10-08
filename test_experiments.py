"""
Tests for mckp_tools.py and the experiment scripts.

The toolkit re-implements the exact solver and greedy on a generic structure, so every
piece is tied back to code that is already verified:
  * parcel-level exact optimum   == solve_exact_front.ExactSolver
  * array greedy                 == baseline_comparison.greedy_baseline (same carbon, same parcels)
  * block gains/weights          == generate_scenarios.score_scenario on the equivalent scenario
  * block size 1                 == the per-parcel problem

    python test_experiments.py
"""
import sys
import tempfile
import os

import numpy as np
import pandas as pd

HINT = ("Run `python check_install.py` to see exactly which files differ from the latest delivery, copy ALL of them "
        "into the same folder, then rerun.")
try:
    import baseline_comparison as bc
    import carbon_time_dynamics as ctd
    import check_greedy_agreement as cga
    import compare_candidate_sets as ccs
    import compare_methods_at_budget as cmb
    import experiment_gain_uncertainty as egu
    import experiment_granularity as egr
    import generate_scenarios as gs
    import mckp_tools as mt
    import nsga2_scenario_search as nss
    import solve_exact_front as sef
    import test_scoring_consistency as tsc
except ImportError as e:
    sys.exit(f"Import failed ({e}). The files in this folder come from different versions.\n{HINT}")


def check_files_current():
    """Fail with a clear message, not a TypeError/AttributeError, when files from different deliveries are mixed."""
    import inspect
    stale = []
    if "fill" not in inspect.signature(bc.greedy_baseline).parameters:
        stale.append("baseline_comparison.py has no skip-and-fill greedy (greedy_baseline has no 'fill' argument)")
    if not hasattr(ctd, "policy_year_stock"):
        stale.append("carbon_time_dynamics.py has no policy_year_stock()")
    if not hasattr(gs, "resolve_reclassify_setting"):
        stale.append("generate_scenarios.py has no reclassification switch (resolve_reclassify_setting)")
    if "tie_priority" not in inspect.signature(mt.solve_greedy).parameters:
        stale.append("mckp_tools.py predates the tie-priority option")
    if not hasattr(cmb, "check_same_candidate_set"):
        stale.append("compare_methods_at_budget.py lacks the candidate-set consistency check")
    if "fill" not in inspect.getsource(cga.agreement):
        stale.append("check_greedy_agreement.py does not cover the fill rule")
    if "policy_year_stock" not in inspect.getsource(nss.build_lookup_tables):
        stale.append("nsga2_scenario_search.py still carries its own copy of the gain formula")
    if stale:
        print("Out-of-date files in this folder:")
        for s in stale:
            print("  - " + s)
        sys.exit(HINT)


check_files_current()

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if not cond and detail else ""))


def close(a, b, rel=1e-6, ab=1e-6):
    return abs(a - b) <= ab + rel * max(abs(a), abs(b))


def make_world(n=80, seed=0):
    combined, ids, means, stds = tsc.synthetic_region(n=n, seed=seed)
    problem = nss.ScenarioProblem(ids, combined, means, stds)
    solver = sef.ExactSolver(problem, nss.evaluate_batch, True)
    return combined, ids, means, stds, problem, solver


def synthetic_edges(n, seed=0):
    rng = np.random.default_rng(seed)
    a = list(range(n - 1)) + list(range(n - 2))
    b = list(range(1, n)) + list(range(2, n))
    extra = rng.integers(0, n, size=(n // 2, 2))
    a += extra[:, 0].tolist(); b += extra[:, 1].tolist()
    return pd.DataFrame({"a": a, "b": b})


BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
combined, ids, means, stds, problem, solver = make_world()
inst = mt.parcel_instance(solver)

# ---- 1. exact == ExactSolver ----
bad = [(d, mt.solve_exact(inst, d)[1], solver.solve(d)["carbon_delta_t"]) for d in BUDGETS
       if not close(mt.solve_exact(inst, d)[1], solver.solve(d)["carbon_delta_t"], rel=2e-4)]
check("mckp exact == ExactSolver at every budget", not bad, bad)

# ---- 2. greedy == baseline_comparison.greedy_baseline ----
bad = []
for d in BUDGETS:
    res, new_classes = bc.greedy_baseline(ids, combined, means, stds, d)
    chosen, gain = mt.solve_greedy(inst, d)
    orig = dict(zip(combined["parcel_id"], combined["land_cover_class"]))
    changed = {p for p in ids if new_classes[p] != orig[p]}
    mine = {ids[inst.group[m]] for m in chosen}
    if not close(gain, res["carbon_delta_vs_baseline_t"]) or changed != mine:
        bad.append((d, gain, res["carbon_delta_vs_baseline_t"], len(changed ^ mine)))
check("array greedy == baseline_comparison.greedy_baseline (same carbon AND same parcels) at every budget", not bad, bad)

# ---- 3. ordering exact >= fill >= stop (parcel level and several block instances) ----
G, W, own = mt.class_gain_tables(solver)
edge_pairs = mt.edges_to_index_pairs(synthetic_edges(len(ids)), ids)
bad = []
rng = np.random.default_rng(3)
for m in (1, 3, 8, 20):
    blocks = [np.array([i]) for i in range(len(ids))] if m == 1 else mt.make_blocks(len(ids), edge_pairs, m, rng)
    bi, _, _ = mt.block_instance(G, W, blocks, solver.total_area)
    for d in BUDGETS:
        _, e = mt.solve_exact(bi, d)
        _, f = mt.solve_greedy(bi, d, fill=True)
        _, s = mt.solve_greedy(bi, d)
        if not (e >= f - 1e-6 and f >= s - 1e-6):
            bad.append((m, d, e, f, s))
check("exact >= greedy-with-fill >= greedy-stop on parcel and block instances", not bad, bad)

# ---- 4. block size 1 == per-parcel problem ----
singles, _, _ = mt.block_instance(G, W, [np.array([i]) for i in range(len(ids))], solver.total_area)
bad = [(d,) for d in BUDGETS if not close(mt.solve_exact(singles, d)[1], mt.solve_exact(inst, d)[1], rel=2e-4)]
check("block instance with singleton blocks has the same optimum as the per-parcel instance", not bad, bad)

# ---- 5. block gains/weights == score_scenario on the equivalent scenario ----
region = combined.copy()
blocks = mt.make_blocks(len(ids), edge_pairs, 6, np.random.default_rng(11))
rng = np.random.default_rng(5)
bad = []
for trial in range(25):
    pick = rng.integers(-1, 3, size=len(blocks))                    # -1 = leave the block alone
    new = dict(zip(combined["parcel_id"], combined["land_cover_class"]))
    gain = weight = 0.0
    for b, c in enumerate(pick):
        if c < 0:
            continue
        for p in blocks[b]:
            new[ids[p]] = mt.REALISTIC[c]
        gain += G[blocks[b], c].sum(); weight += W[blocks[b], c].sum()
    r = tsc.score(ids, new, region, means, stds)
    if not close(r["carbon_delta_vs_baseline_t"], gain) or not close(r["disruption_fraction"], weight / solver.total_area):
        bad.append((trial, r["carbon_delta_vs_baseline_t"], gain, r["disruption_fraction"], weight / solver.total_area))
check("block gain and weight == score_scenario's carbon and disruption for the equivalent whole-region scenario (25 random block assignments)", not bad, bad[:3])

bi, mb, mc = mt.block_instance(G, W, blocks, solver.total_area)
want = {(b, c): (G[blocks[b], c].sum(), W[blocks[b], c].sum()) for b in range(len(blocks)) for c in range(3)
        if G[blocks[b], c].sum() > 0 and W[blocks[b], c].sum() > 0}
got = {(int(b), int(c)): (g, w) for b, c, g, w in zip(mb, mc, bi.gain, bi.weight)}
check("block_instance contains exactly the positive-gain, positive-weight moves with the right values",
      set(want) == set(got) and all(close(want[k][0], got[k][0]) and close(want[k][1], got[k][1]) for k in want))

# ---- 6. make_blocks ----
bad = []
nbrs = [set() for _ in ids]
for i, j in edge_pairs:
    nbrs[i].add(j); nbrs[j].add(i)
for m in (2, 5, 12):
    bl = mt.make_blocks(len(ids), edge_pairs, m, np.random.default_rng(m))
    flat = np.concatenate(bl)
    if sorted(flat.tolist()) != list(range(len(ids))):
        bad.append((m, "not a partition"))
    for b in bl:
        if len(b) > m:
            bad.append((m, "block too large", len(b)))
        if len(b) > 1:
            seen, stack, s = {int(b[0])}, [int(b[0])], set(map(int, b))
            while stack:
                u = stack.pop()
                for v in nbrs[u] & s:
                    if v not in seen:
                        seen.add(v); stack.append(v)
            if seen != s:
                bad.append((m, "block not connected", sorted(s)))
check("make_blocks: a partition, no block larger than m, every multi-parcel block spatially connected", not bad, bad[:3])

# ---- 7. noise ----
rng = np.random.default_rng(0)
check("noise: sigma=0 leaves gains untouched", np.all(egu.noise_factors(inst, 0.0, rng, "parcel") == 1.0))
for mode in ("parcel", "move"):
    f = np.concatenate([egu.noise_factors(inst, 0.3, rng, mode) for _ in range(400)])
    check(f"noise ({mode}): mean-preserving (E[factor] = 1)", abs(f.mean() - 1) < 0.01, f.mean())
f = egu.noise_factors(inst, 0.3, np.random.default_rng(1), "parcel")
same = all(np.ptp(f[inst.group == g]) < 1e-12 for g in np.unique(inst.group))
check("noise (parcel mode): every move of a parcel shares one factor", same)

# ---- 8. experiment: gain uncertainty ----
raw = egu.run(inst, [0.0, 0.2], draws=6, budgets=[0.1, 0.3, 0.5], seed=2, progress=False)
df, table = egu.summarise(raw)
z = df[df["sigma"] == 0.0]
check("uncertainty experiment, sigma=0: exact plan realises 100% and greedy matches the unperturbed ratio",
      np.allclose(z["exact_plan_pct"], 100.0, atol=0.02) and
      all(abs(r["greedy_plan_pct"] - 100 * mt.solve_greedy(inst, r["budget"], fill=True)[1] / mt.solve_exact(inst, r["budget"])[1]) < 0.02 for _, r in z.iterrows()))
raw_stop = egu.run(inst, [0.0], draws=2, budgets=[0.1, 0.3], seed=2, progress=False, greedy_fill=False)
_, tab_stop = egu.summarise(raw_stop)
check("uncertainty experiment: greedy_fill=False uses the stop rule (sigma=0 reproduces solve_greedy without fill)",
      all(abs(r["greedy_informed"] - mt.solve_greedy(inst, r["budget"])[1]) < 1e-6 for _, r in raw_stop.iterrows()))
check("uncertainty experiment: the fill rule is never worse than the stop rule at sigma=0",
      all(raw[(raw.sigma == 0.0) & (raw.budget == b)]["greedy_informed"].iloc[0] >= raw_stop[raw_stop.budget == b]["greedy_informed"].iloc[0] - 1e-6 for b in (0.1, 0.3)))
check("uncertainty experiment: no plan ever beats the perfectly-informed optimum",
      (df["greedy_informed_pct"] <= 100.02).all() and (df["exact_plan_pct"] <= 100.02).all() and (df["greedy_plan_pct"] <= 100.02).all())
with tempfile.TemporaryDirectory() as td:
    egu.plot(df, os.path.join(td, "u.png"))
    check("uncertainty experiment: figure is written", os.path.exists(os.path.join(td, "u.png")))

# ---- 9. experiment: granularity ----
gdf = egr.run(solver, edge_pairs, sizes=[1, 4, 12], partitions=3, budgets=[0.1, 0.3, 0.5], seed=4)
one = gdf[gdf["block_size"] == 1]
check("granularity experiment, block size 1: exact optimum == ExactSolver at each budget",
      all(close(r["opt"], solver.solve(r["budget"])["carbon_delta_t"], rel=2e-4) for _, r in one.iterrows()))
feas = gdf[gdf["opt"] > 0]
check("granularity experiment: gaps are NaN exactly when nothing fits (opt = 0), otherwise >= 0 with 'fill' never worse than 'stop'",
      gdf.loc[gdf["opt"] == 0, "gap_stop_pct"].isna().all() and feas["gap_stop_pct"].notna().all()
      and (feas["gap_stop_pct"] >= -1e-3).all() and (feas["gap_fill_pct"] >= -1e-3).all()
      and (feas["gap_fill_pct"] <= feas["gap_stop_pct"] + 1e-6).all())
check("granularity experiment: the 'fits' flag matches opt > 0", (gdf["fits"] == (gdf["opt"] > 0)).all())
print("      (info, not asserted: mean 'stop' gap by block size on this synthetic region -> "
      + ", ".join(f"{m}: {g:.2f}%" for m, g in gdf.groupby("block_size")["gap_stop_pct"].mean().items()) + ")")
with tempfile.TemporaryDirectory() as td:
    egr.plot(gdf, os.path.join(td, "g.png"))
    check("granularity experiment: figure and summary are produced", os.path.exists(os.path.join(td, "g.png")) and len(egr.summarise(gdf)) == 3)

# ---- 10. experiment: candidate sets ----
keep = [p for p in ids if p % 7 != 0]
pa = nss.ScenarioProblem(keep, combined[combined.parcel_id.isin(keep)].copy(), means, stds)
table, new_ids, sb, ib = ccs.run(pa, problem, nss.evaluate_batch, budgets=[0.1, 0.3, 0.5])
check("candidate-set experiment: adding candidates can never lower the optimum (same fraction OR same hectares)",
      ((table["optimum_with_added"] - table["optimum_now"]) >= -1e-3).all()
      and ((table["optimum_with_added_same_ha"] - table["optimum_now"]) >= -1e-3).all(),
      table[["optimum_now", "optimum_with_added", "optimum_with_added_same_ha"]].values.tolist())
check("candidate-set experiment: the same-hectares value never exceeds the same-fraction value (a larger budget can only help)",
      (table["optimum_with_added_same_ha"] <= table["optimum_with_added"] + 1e-3).all())
check("candidate-set experiment: reports exactly the added parcels", new_ids == set(ids) - set(keep) and "parcels added" in ccs.describe_new(problem, sb, ib, new_ids))

# ---- 11. reclassification ----
c = pd.DataFrame({"land_cover_class": ["unresolved_crop"] * 5 + ["grassland", "excluded"],
                  "crop": ["MAIZE", "kale ", "POTATOES", "FODDER BEET", "SUGAR BEET", "MAIZE", "SCRUB"],
                  "area_ha": 1.0})
out = gs.reclassify_unresolved_tillage_like(c.copy(), verbose=False)
check("reclassify: only whitelisted crops among UNRESOLVED parcels become tillage (case/space-insensitive)",
      out["land_cover_class"].tolist() == ["tillage", "tillage", "unresolved_crop", "tillage", "unresolved_crop", "grassland", "excluded"],
      out["land_cover_class"].tolist())
again = gs.reclassify_unresolved_tillage_like(out.copy(), verbose=False)
check("reclassify: idempotent", again["land_cover_class"].tolist() == out["land_cover_class"].tolist())
custom = gs.reclassify_unresolved_tillage_like(c.copy(), crops=["potatoes"], verbose=False)
check("reclassify: the crop list is configurable", custom["land_cover_class"].tolist()[2] == "tillage")
check("reclassify: the switch is a plain bool (its shipped value is a project decision, not tested here)", isinstance(gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED, bool))

# ---- 12. TIES: real baselines come from soil-association values, so many parcels tie exactly ----
def tied_world(n=300, seed=0, n_values=5):
    rng = np.random.default_rng(seed)
    cls = rng.choice(["tillage", "grassland", "forestry", "peatland"], n, p=[.25, .45, .15, .15])
    vals = {"tillage": np.linspace(110, 250, n_values), "grassland": np.linspace(200, 400, n_values),
            "forestry": np.array([448.1]), "peatland": np.array([705.0])}
    comb = pd.DataFrame({"parcel_id": np.arange(n), "area_ha": rng.lognormal(1.0, 0.8, n), "land_cover_class": cls,
                         "carbon_t_c_per_ha_mean": [rng.choice(vals[c]) for c in cls],
                         "grassland_index_mean": [1.0 if c == "grassland" else np.nan for c in cls], "crop": "X"})
    mns = comb.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].mean().to_dict()
    sds = {k: 1.0 for k in mns}
    pids = list(range(n))
    prob = nss.ScenarioProblem(pids, comb, mns, sds)
    return comb, pids, mns, sds, prob, sef.ExactSolver(prob, nss.evaluate_batch, True)

bad = []
for nv in (3, 5, 12):
    c2, p2, m2, s2, pr2, sv2 = tied_world(n_values=nv)
    in2 = mt.parcel_instance(sv2)
    orig2 = dict(zip(c2.parcel_id, c2.land_cover_class))
    for d in BUDGETS:
        res, new = bc.greedy_baseline(p2, c2, m2, s2, d)
        a_set = {p for p in p2 if new[p] != orig2[p]}
        ch, g = mt.solve_greedy(in2, d)
        b_set = {p2[in2.group[m]] for m in ch}
        if a_set != b_set or not close(res["carbon_delta_vs_baseline_t"], g):
            bad.append((nv, d, res["carbon_delta_vs_baseline_t"], g, len(a_set ^ b_set)))
check("TIED baselines: baseline_comparison.greedy_baseline and mckp greedy pick the same parcels (3, 5 and 12 distinct values)", not bad, bad[:4])

c2, p2, m2, s2, pr2, sv2 = tied_world(n_values=5)
in2 = mt.parcel_instance(sv2)
import check_greedy_agreement as cga
tp = cga.tie_prevalence(in2)
check("tie diagnostic: detects that many parcels share their density", tp["parcels_sharing_density"] > 0.5 * tp["positive_gain_parcels"], tp)
ts = cga.tie_sensitivity(sv2, in2, budgets=[0.2, 0.4], n_random=8)
check("tie diagnostic: default tie order lies inside the [min, max] over tie orders, and fill-greedy never below stop-greedy",
      ((ts["min"] <= ts["default"] + 1e-6) & (ts["default"] <= ts["max"] + 1e-6)).all()
      and (ts[ts.rule == "fill"].set_index("budget")["default"] >= ts[ts.rule == "stop"].set_index("budget")["default"] - 1e-6).all())
check("tie diagnostic: on a TIED region the tie order really moves greedy (so the diagnostic has something to report)",
      (ts[ts.rule == "stop"]["range_%_of_max"] > 0).any(), ts[["budget", "rule", "range_%_of_max"]].values.tolist())
ag = cga.agreement(pr2, sv2, in2, budgets=[0.2, 0.4])
check("agreement diagnostic: reports zero differing parcels on the tied region", (ag["parcels_that_differ"] == 0).all(), ag.to_dict("records"))

# ---- 13. heteroscedastic (per-parcel) sigma ----
rng = np.random.default_rng(0)
sg = np.where(np.arange(inst.n_groups) % 2 == 0, 0.05, 0.4)
fs = np.array([egu.noise_factors(inst, sg, rng, "parcel") for _ in range(600)])
per_parcel = np.array([fs[:, inst.group == g][:, 0] for g in np.unique(inst.group)]).T     # draws x parcels
sds = per_parcel.std(axis=0)
low, high = sds[np.unique(inst.group) % 2 == 0], sds[np.unique(inst.group) % 2 == 1]
check("per-parcel sigma: noise is larger for parcels with larger sigma, and still mean-preserving",
      high.mean() > 3 * low.mean() and abs(per_parcel.mean() - 1) < 0.02, (low.mean(), high.mean(), per_parcel.mean()))
raw2 = egu.run(inst, [(0.2, sg)], draws=3, budgets=[0.2, 0.4], seed=1, progress=False)
check("per-parcel sigma: run() accepts a (label, array) item and labels it", set(raw2["sigma"]) == {0.2} and len(raw2) == 6)

# ---- 14. how the reclassification switch is resolved ----
import os as _os
import compare_methods_at_budget as cmb

_saved_env = _os.environ.pop("RECLASSIFY_TILLAGE_LIKE_UNRESOLVED", None)
_saved_const = gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED
try:
    gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED = False
    check("switch: with no env var, the constant in generate_scenarios.py decides",
          gs.resolve_reclassify_setting()[0] is False and "default" in gs.resolve_reclassify_setting()[1])
    gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED = True
    check("switch: flipping the constant to True turns it on for every caller that passes no argument",
          gs.resolve_reclassify_setting() [0] is True)
    gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED = False
    ok = True
    for val, want in (("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("false", False), ("no", False)):
        _os.environ["RECLASSIFY_TILLAGE_LIKE_UNRESOLVED"] = val
        got, where = gs.resolve_reclassify_setting()
        ok &= (got is want and "environment" in where)
    check("switch: the environment variable overrides the constant (1/true/yes/on -> True; 0/false/no -> False)", ok)
    _os.environ["RECLASSIFY_TILLAGE_LIKE_UNRESOLVED"] = ""
    check("switch: an empty environment variable is ignored", gs.resolve_reclassify_setting()[0] is False)
    gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED = True
    _os.environ["RECLASSIFY_TILLAGE_LIKE_UNRESOLVED"] = "0"
    check("switch: env var beats the constant, and an explicit argument beats both",
          gs.resolve_reclassify_setting()[0] is False and gs.resolve_reclassify_setting(True)[0] is True
          and gs.resolve_reclassify_setting(False)[0] is False and "explicit" in gs.resolve_reclassify_setting(True)[1])
finally:
    gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED = _saved_const
    _os.environ.pop("RECLASSIFY_TILLAGE_LIKE_UNRESOLVED", None)
    if _saved_env is not None:
        _os.environ["RECLASSIFY_TILLAGE_LIKE_UNRESOLVED"] = _saved_env

# ---- 15. exports built on different candidate sets must not be compared ----
def fake_export(path, parcel_ids, method, scenario_id, carbon):
    pd.DataFrame({"method": method, "scenario_id": scenario_id, "disruption_fraction": 0.1,
                  "carbon_delta_vs_baseline_t": carbon, "parcel_id": list(parcel_ids)}).to_parquet(path)

with tempfile.TemporaryDirectory() as td:
    same = list(range(597))
    fake_export(os.path.join(td, "scenario_details_seed1.parquet"), same, "nsga2", "seed1_0", 100.0)
    fake_export(os.path.join(td, "scenario_details_seed2.parquet"), same, "nsga2", "seed2_0", 101.0)
    base = os.path.join(td, "base.parquet")
    fake_export(base, same, "greedy", "0.1", 99.0)
    sc = cmb.load_scenarios(os.path.join(td, "scenario_details_seed*.parquet"), base)
    check("compare_methods_at_budget: exports on the SAME candidate set load fine", len(sc) == 3)

    fake_export(base, list(range(611)), "greedy", "0.1", 99.0)          # baselines run WITH the reclassification, NSGA-II without
    try:
        cmb.load_scenarios(os.path.join(td, "scenario_details_seed*.parquet"), base); refused, msg = False, ""
    except SystemExit as e:
        refused, msg = True, str(e)
    check("compare_methods_at_budget: baselines on 611 parcels vs NSGA-II on 597 is refused, naming the files and counts",
          refused and "DIFFERENT candidate sets" in msg and "611" in msg and "597" in msg and "base.parquet" in msg, msg[:150])

# ---- 16. the real loader: banner printed, reclassification actually lands in the loaded data ----
import contextlib
import io
import geopandas as gpd
from shapely.geometry import box

_rows = [  # tillage_class, is_grassland, carbon_source, is_excluded, crop
    ("WINTER WHEAT", False, "sis_measured_series", False, "WINTER WHEAT"), (None, True, "sis_measured_series", False, "PASTURE"),
    (None, False, "nfi_blended_average", False, "FORESTRY"), (None, False, None, True, "SCRUB"),
    (None, False, "sis_measured_series", False, "MAIZE"), (None, False, "sis_measured_series", False, "FODDER BEET"),
    (None, False, "sis_measured_series", False, "POTATOES - MAINCROP")]
_gdf = gpd.GeoDataFrame({"tillage_class": [r[0] for r in _rows], "is_grassland": [r[1] for r in _rows], "carbon_source": [r[2] for r in _rows],
                         "is_excluded": [r[3] for r in _rows], "crop": [r[4] for r in _rows],
                         "carbon_t_c_per_ha_mean": [150, 300, 440, 0, 120, 110, 130]},
                        geometry=[box(i * 200, 0, i * 200 + 100, 100) for i in range(len(_rows))], crs="EPSG:2157")
_orig_read, _orig_env, _orig_const = gs.gpd.read_file, _os.environ.pop("RECLASSIFY_TILLAGE_LIKE_UNRESOLVED", None), gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED
gs.gpd.read_file = lambda path: _gdf.copy()


def _load(env=None, arg=None):
    _os.environ.pop("RECLASSIFY_TILLAGE_LIKE_UNRESOLVED", None)
    if env is not None:
        _os.environ["RECLASSIFY_TILLAGE_LIKE_UNRESOLVED"] = env
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c = gs.load_area_and_baseline(arg)
    return c["land_cover_class"].tolist(), buf.getvalue()


try:
    gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED = False
    cls, out = _load()
    check("loader, switch off: nothing reclassified; the banner says so and names its source",
          cls.count("unresolved_crop") == 3 and "NOT reclassified" in out and "default in generate_scenarios.py" in out, (cls, out))
    cls, out = _load(env="1")
    check("loader, RECLASSIFY_TILLAGE_LIKE_UNRESOLVED=1: maize and fodder beet become tillage, potatoes stay unresolved",
          cls == ["tillage", "grassland", "forestry", "excluded", "tillage", "tillage", "unresolved_crop"] and "RECLASSIFIED" in out and "environment variable" in out, (cls, out))
    cls, out = _load(env="0", arg=True)
    check("loader: an explicit argument beats the environment variable", cls.count("tillage") == 3 and "explicit argument" in out, (cls, out))
    gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED = True
    cls, out = _load()
    check("loader: editing the constant to True in generate_scenarios.py is enough",
          cls.count("tillage") == 3 and "default in generate_scenarios.py" in out and "RECLASSIFIED" in out, (cls, out))
finally:
    gs.gpd.read_file, gs.RECLASSIFY_TILLAGE_LIKE_UNRESOLVED = _orig_read, _orig_const
    _os.environ.pop("RECLASSIFY_TILLAGE_LIKE_UNRESOLVED", None)
    if _orig_env is not None:
        _os.environ["RECLASSIFY_TILLAGE_LIKE_UNRESOLVED"] = _orig_env

# ---- 17. the skip-and-fill greedy ----
bad = []
for label, (cx, ix, mx, sx, px, sv) in {"untied": (combined, ids, means, stds, problem, solver), "tied": tied_world(n_values=5)}.items():
    ins = mt.parcel_instance(sv)
    og = dict(zip(cx.parcel_id, cx.land_cover_class))
    for d in BUDGETS:
        res, new = bc.greedy_baseline(ix, cx, mx, sx, d, fill=True)
        a_set = {p for p in ix if new[p] != og[p]}
        ch, g = mt.solve_greedy(ins, d, fill=True)
        b_set = {ix[ins.group[m]] for m in ch}
        res_stop, _ = bc.greedy_baseline(ix, cx, mx, sx, d)
        _, opt = mt.solve_exact(ins, d)
        if a_set != b_set or not close(res["carbon_delta_vs_baseline_t"], g):
            bad.append((label, d, "fill differs between baseline_comparison and mckp", len(a_set ^ b_set)))
        if res["carbon_delta_vs_baseline_t"] < res_stop["carbon_delta_vs_baseline_t"] - 1e-6:
            bad.append((label, d, "fill below stop"))
        if res["carbon_delta_vs_baseline_t"] > opt * (1 + 2e-4) + 1e-6:
            bad.append((label, d, "fill above the exact optimum"))
        if res["disruption_fraction"] > d + 1e-9:
            bad.append((label, d, "fill exceeds the budget", res["disruption_fraction"]))
check("fill-greedy: baseline_comparison and mckp pick the same parcels; fill >= stop; fill <= exact; within budget (tied and untied regions)", not bad, bad[:4])

import compare_methods_at_budget as cmb2
with tempfile.TemporaryDirectory() as td:
    out_path = os.path.join(td, "base.parquet")
    det = bc.export_scenario_details(ids, combined, means, stds, n_random_seeds=2, skip_filter=True, out_path=out_path)
    table = bc.run_comparison(ids, n_random_seeds=2, combined=combined, branch_means=means, branch_stds=stds, skip_filter=True)
    check("export carries method 'greedy_fill' as well as 'greedy' and 'random'", {"greedy", "greedy_fill", "random"} <= set(det["method"]))
    check("run_comparison prints a greedy_fill_heuristic row for every budget", (table["method"] == "greedy_fill_heuristic").sum() == len(bc.TARGET_DISRUPTION_LEVELS))
    f = det[det.method == "greedy_fill"].drop_duplicates("scenario_id").set_index("scenario_id")["carbon_delta_vs_baseline_t"]
    t_ = table[table.method == "greedy_fill_heuristic"].set_index("target_disruption")["carbon_delta_t"]
    check("the exported fill-greedy scenario carbon equals the comparison table's", all(close(f[f"{d:.1f}"], t_[d]) for d in bc.TARGET_DISRUPTION_LEVELS))
    # NSGA-II-style export + the comparison tool
    nrows = []
    for s in (1, 2):
        for k in range(3):
            for p in ids:
                nrows.append({"method": "nsga2", "scenario_id": f"seed{s}_{k}", "disruption_fraction": 0.05 + 0.1 * k + 0.01 * s,
                              "carbon_delta_vs_baseline_t": 1000.0 * (k + 1) + s, "parcel_id": p})
    pd.DataFrame(nrows[: len(nrows) // 2]).to_parquet(os.path.join(td, "scenario_details_seed1.parquet"))
    pd.DataFrame(nrows[len(nrows) // 2:]).to_parquet(os.path.join(td, "scenario_details_seed2.parquet"))
    tab, _, _ = cmb2.compare(cmb2.load_scenarios(os.path.join(td, "scenario_details_seed*.parquet"), out_path))
    check("compare_methods_at_budget reports fill-greedy and NSGA-II relative to it",
          tab["greedy_fill_carbon"].notna().all() and tab["nsga2_best_pct_vs_greedy_fill"].notna().all()
          and all(close(tab.loc[i, "nsga2_best_pct_vs_greedy_fill"], 100 * (tab.loc[i, "nsga2_best_seed_carbon"] / tab.loc[i, "greedy_fill_carbon"] - 1)) for i in tab.index))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
