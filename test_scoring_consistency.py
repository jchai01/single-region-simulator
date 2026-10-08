"""
Scoring-consistency test.

The question "how much carbon does this move earn?" is answered in three
places that must agree EXACTLY:
  1. generate_scenarios.score_scenario()        -- scores greedy and random
  2. nsga2_scenario_search.build_lookup_tables() -- the table NSGA-II (and the
                                                   exact solver) optimise on
  3. baseline_comparison.best_realistic_target() -- how greedy RANKS parcels

When the carbon accounting changed (time dynamics, then the SOLUM hybrid), the
scorer and the lookup table were updated but greedy's ranking was not, TWICE.
Greedy then ranked parcels by one objective and was scored on another -- close
to the reverse order -- and NSGA-II appeared to beat it by 45-120%. This test
makes that class of bug impossible to miss.

    python test_scoring_consistency.py            # synthetic region; runs anywhere
    python test_scoring_consistency.py --real     # also checks YOUR region (~1-2 min)
    python test_scoring_consistency.py --real --all   # every parcel (slower)

Exit code 0 = everything agrees; 1 = something drifted (details printed).
"""
import argparse
import sys

import numpy as np
import pandas as pd

FILES = "carbon_time_dynamics.py, generate_scenarios.py, baseline_comparison.py, nsga2_scenario_search.py and test_scoring_consistency.py (run `python check_install.py` to see exactly which differ)"
try:
    import baseline_comparison as bc
    import carbon_time_dynamics as ctd
    import generate_scenarios as gs
    import nsga2_scenario_search as nss
except ImportError as e:
    sys.exit(f"Import failed ({e}).\nThe files in this folder come from different versions. Copy ALL of {FILES} "
             "from the latest delivery into the same folder, then rerun.")


def check_files_current():
    """Fail with a clear message, not an AttributeError, if some files predate the shared-formula refactor."""
    import inspect
    stale = []
    if not hasattr(ctd, "policy_year_stock"):
        stale.append("carbon_time_dynamics.py has no policy_year_stock()")
    if not hasattr(bc, "best_realistic_target"):
        stale.append("baseline_comparison.py is the OLD version (greedy's ranking is still an inner function nothing can test)")
    if "exclude_missing_carbon" not in inspect.signature(gs.filter_reallocatable).parameters:
        stale.append("generate_scenarios.py predates the missing-carbon exclusion")
    if "policy_year_stock" not in inspect.getsource(nss.build_lookup_tables):
        stale.append("nsga2_scenario_search.py still carries its own copy of the gain formula (not using policy_year_stock)")
    if stale:
        print("Out-of-date files in this folder:")
        for s in stale:
            print("  - " + s)
        sys.exit(f"Copy ALL of {FILES} from the latest delivery into the same folder, then rerun.")

RTOL, ATOL = 1e-9, 1e-6


def close(a, b):
    return abs(a - b) <= ATOL + RTOL * max(abs(a), abs(b))


class Report:
    def __init__(self):
        self.results = []

    def check(self, name, ok, detail=""):
        self.results.append(bool(ok))
        print(("PASS  " if ok else "FAIL  ") + name)
        if not ok and detail:
            for line in (detail if isinstance(detail, list) else [detail])[:6]:
                print("        " + str(line))

    @property
    def ok(self):
        return all(self.results)


def synthetic_region(n=70, seed=0):
    rng = np.random.default_rng(seed)
    cls = rng.choice(["tillage", "grassland", "forestry", "peatland"], n, p=[.25, .42, .15, .18])
    lo_hi = {"tillage": (100, 260), "grassland": (180, 420), "forestry": (300, 520), "peatland": (600, 780)}
    combined = pd.DataFrame({
        "parcel_id": np.arange(n), "area_ha": rng.lognormal(1.0, 0.8, n), "land_cover_class": cls,
        "carbon_t_c_per_ha_mean": [rng.uniform(*lo_hi[c]) for c in cls],
        "grassland_index_mean": [1.0 if c == "grassland" else np.nan for c in cls],
        "crop": ["X"] * n,
    })
    means = combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].mean().to_dict()
    stds = combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].std().to_dict()
    return combined, list(range(n)), means, stds


def score(parcel_ids, new_classes, region, means, stds):
    return gs.score_scenario(parcel_ids, new_classes, region, means, stds, 0.0, 0.0, 0.0)


def compare_paths(rep, label, combined, parcel_ids, means, stds, greedy_fn, sample=None, expect_failure=False):
    """Checks all three paths against each other. Returns the list of mismatches."""
    region = combined[combined["parcel_id"].isin(parcel_ids)].copy()
    problem = nss.ScenarioProblem(parcel_ids, region, means, stds)
    t, labels = problem._tables, problem.per_parcel_labels
    area, lookup = np.asarray(t["area_ha"]), np.asarray(t["padded_carbon_lookup"])
    total_area = float(t["total_area_ha"])
    reg = region.set_index("parcel_id").loc[parcel_ids]
    base_ha, cls0 = reg["carbon_t_c_per_ha_mean"].to_numpy(), reg["land_cover_class"].tolist()
    original = dict(zip(parcel_ids, cls0))
    n = len(parcel_ids)
    rng = np.random.default_rng(7)
    idxs = list(range(n)) if sample is None or sample >= n else sorted(rng.choice(n, sample, replace=False).tolist())

    # 1. NSGA-II lookup table vs score_scenario, for every single move
    bad = []
    for p in idxs:
        pid = parcel_ids[p]
        for k, lab in enumerate(labels[p]):
            new = dict(original); new[pid] = lab
            r = score(parcel_ids, new, region, means, stds)
            want_c = area[p] * (lookup[p, k] - base_ha[p])
            want_d = 0.0 if lab == original[pid] else area[p] / total_area
            if not close(r["carbon_delta_vs_baseline_t"], want_c):
                bad.append(f"parcel {pid} {original[pid]}->{lab}: score_scenario {r['carbon_delta_vs_baseline_t']:.4f} t C vs lookup {want_c:.4f}")
            if not close(r["disruption_fraction"], want_d):
                bad.append(f"parcel {pid} {original[pid]}->{lab}: disruption {r['disruption_fraction']:.6f} vs {want_d:.6f}")
    if not expect_failure:
        rep.check(f"[{label}] NSGA-II lookup table == score_scenario on every single move ({len(idxs)} parcels)", not bad, bad)

    # 2. greedy's ranking vs the lookup table and vs score_scenario
    bad2 = []
    for p in idxs:
        cur = cls0[p]
        cls, gain = greedy_fn(cur, base_ha[p], means)
        cands = {lab: lookup[p, k] - base_ha[p] for k, lab in enumerate(labels[p])
                 if lab in bc.REALISTIC_TARGET_CLASSES and lab != cur}
        if not cands:
            continue
        best_gain = max(cands.values())
        if cls not in cands or not close(cands[cls], best_gain) or not close(gain, best_gain):
            bad2.append(f"parcel {parcel_ids[p]} ({cur}): greedy picks {cls} (claims {gain:.3f} t/ha); "
                        f"lookup says best is {max(cands, key=cands.get)} ({best_gain:.3f} t/ha), "
                        f"and {cls} is actually worth {cands.get(cls, float('nan')):.3f}")
            continue
        new = dict(original); new[parcel_ids[p]] = cls
        r = score(parcel_ids, new, region, means, stds)
        if not close(r["carbon_delta_vs_baseline_t"], gain * area[p]):
            bad2.append(f"parcel {parcel_ids[p]}: greedy's claimed gain x area {gain * area[p]:.4f} vs score_scenario {r['carbon_delta_vs_baseline_t']:.4f}")
    if not expect_failure:
        rep.check(f"[{label}] greedy's ranking gain == lookup table == score_scenario for its chosen move", not bad2, bad2)

    # 3. whole scenarios: evaluate_batch vs score_scenario
    if not expect_failure:
        X = np.column_stack([rng.integers(0, len(labels[p]), size=25) for p in range(n)])
        F = nss.evaluate_batch(X, t)
        bad3 = []
        for i in range(X.shape[0]):
            new = {pid: labels[p][X[i, p]] for p, pid in enumerate(parcel_ids)}
            r = score(parcel_ids, new, region, means, stds)
            if not close(r["carbon_delta_vs_baseline_t"], -F[i, 0]):
                bad3.append(f"scenario {i}: carbon score_scenario {r['carbon_delta_vs_baseline_t']:.4f} vs evaluate_batch {-F[i, 0]:.4f}")
            if not close(r["disruption_fraction"], F[i, 1]):
                bad3.append(f"scenario {i}: disruption {r['disruption_fraction']:.6f} vs {F[i, 1]:.6f}")
        rep.check(f"[{label}] 25 random whole scenarios: evaluate_batch == score_scenario (carbon and disruption)", not bad3, bad3)
    return bad + bad2


def flat_target_greedy(current_class, own_baseline, branch_means):
    """The OLD, buggy ranking: flat branch-mean target instead of the SOLUM hybrid."""
    gains = {c: ctd.carbon_at_year(current_class, c, own_baseline, branch_means[c], ctd.YEARS_TO_POLICY_TARGET) - own_baseline
             for c in bc.REALISTIC_TARGET_CLASSES if c != current_class}
    if not gains:
        return current_class, 0.0
    best = max(gains, key=gains.get)
    return best, gains[best]


def test_constants_and_helpers(rep):
    bad = [f"{m.__name__}.YEARS_TO_POLICY_TARGET = {getattr(m, 'YEARS_TO_POLICY_TARGET')} != {ctd.YEARS_TO_POLICY_TARGET}"
           for m in (gs, bc, nss) if hasattr(m, "YEARS_TO_POLICY_TARGET")
           and getattr(m, "YEARS_TO_POLICY_TARGET") != ctd.YEARS_TO_POLICY_TARGET]
    rep.check("one policy horizon: every module uses carbon_time_dynamics.YEARS_TO_POLICY_TARGET", not bad, bad)


def test_filter(rep):
    rows = [("tillage", "WHEAT", 150.0), ("grassland", "PASTURE", 300.0), ("forestry", "FORESTRY", 440.0),
            ("excluded", "SCRUB", 0.0), ("unresolved_crop", "MAIZE", 120.0), ("grassland", "PASTURE", np.nan),
            ("grassland", "BUILDING", 300.0)]
    comb = pd.DataFrame({"parcel_id": range(len(rows)), "land_cover_class": [r[0] for r in rows], "crop": [r[1] for r in rows],
                         "carbon_t_c_per_ha_mean": [r[2] for r in rows], "area_ha": 1.0})
    saved, gs.flag_likely_commonage = gs.flag_likely_commonage, (lambda *a, **k: set())
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            kept = gs.filter_reallocatable(list(range(len(rows))), comb)
            kept_nan_ok = gs.filter_reallocatable(list(range(len(rows))), comb, exclude_missing_carbon=False)
    finally:
        gs.flag_likely_commonage = saved
    rep.check("filter_reallocatable keeps only real candidates (drops immutable, excluded, unresolved_crop, NaN-carbon)", kept == [0, 1, 2], kept)
    rep.check("...and exclude_missing_carbon=False restores the NaN-carbon parcel", 5 in kept_nan_ok, kept_nan_ok)


def test_random_uses_its_budget(rep):
    combined, ids, means, stds = synthetic_region(n=120, seed=3)
    region = combined
    max_share = float(region["area_ha"].max() / region["area_ha"].sum())
    bad = []
    for target in (0.2, 0.4):
        for seed in range(4):
            res, new = bc.random_baseline(ids, combined, means, stds, target, seed=seed)
            orig = dict(zip(combined["parcel_id"], combined["land_cover_class"]))
            if any(new[p] != orig[p] and new[p] not in bc.REALISTIC_TARGET_CLASSES for p in ids):
                bad.append("random assigned a non-realistic class")
            d = res["disruption_fraction"]
            if not (target - max_share - 1e-9 <= d <= target + 1e-9):
                bad.append(f"target {target} seed {seed}: realised disruption {d:.3f} (should be within one parcel of the target)")
    rep.check("random baseline realises its disruption budget (no 'reallocations' to a parcel's own class)", not bad, bad)


def run_real(rep, sample):
    combined = gs.load_area_and_baseline()
    with open("data/offaly_subregion_parcel_ids.txt") as f:
        ids = [int(line.strip()) for line in f if line.strip()]
    parcel_ids = gs.filter_reallocatable(ids, combined)
    means = bc.branch_means_from_loaded(combined)
    stds = bc.branch_stds_from_loaded(combined)
    print(f"\nREAL region: {len(parcel_ids)} candidate parcels")
    carbon_by_pid = dict(zip(combined["parcel_id"], combined["carbon_t_c_per_ha_mean"]))
    rep.check("[real] no candidate parcel has a NaN carbon baseline", not any(pd.isna(carbon_by_pid[p]) for p in parcel_ids))
    try:
        import solve_exact_front as sef
        problem = sef.build_problem()
        rep.check("[real] greedy/random and the search/solver use the SAME candidate parcels",
                  sorted(problem.parcel_ids) == sorted(parcel_ids),
                  [f"baselines use {len(parcel_ids)} parcels, search/solver {len(problem.parcel_ids)}"])
    except Exception as e:  # noqa: BLE001
        print(f"(skipped candidate-set comparison with the solver's problem: {type(e).__name__}: {e})")
    compare_paths(rep, "real", combined, parcel_ids, means, stds, bc.best_realistic_target, sample=sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="also check the real case-study region")
    ap.add_argument("--all", action="store_true", help="with --real: check every parcel, not a sample")
    ap.add_argument("--sample", type=int, default=150, help="with --real: number of parcels to sample")
    args = ap.parse_args()

    check_files_current()
    rep = Report()
    print("--- synthetic region ---")
    combined, ids, means, stds = synthetic_region()
    compare_paths(rep, "synthetic", combined, ids, means, stds, bc.best_realistic_target)

    # The test must be able to FAIL: the old flat-target ranking has to be detected.
    caught = compare_paths(rep, "self-check", combined, ids, means, stds, flat_target_greedy, expect_failure=True)
    rep.check("self-check: this test DOES detect the old stale-ranking bug (flat-target greedy)", len(caught) > 0,
              "the checker would have missed the bug that distorted the earlier results")

    test_constants_and_helpers(rep)
    test_filter(rep)
    test_random_uses_its_budget(rep)
    if args.real:
        run_real(rep, None if args.all else args.sample)

    print(f"\n{sum(rep.results)}/{len(rep.results)} checks passed")
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
