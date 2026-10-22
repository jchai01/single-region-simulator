"""Tests for coupled_objective.py: brute force on tiny instances, vectorised-vs-reference boundary,
and consistency with the verified additive code. Run: python test_coupled.py"""
import itertools, sys
import numpy as np
import test_scoring_consistency as tsc
import nsga2_scenario_search as nss
import solve_exact_front as sef
import mckp_tools as mt
import coupled_objective as co

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond: fails.append(name)

def world(n, seed, p_edge=0.35):
    c, ids, m, s = tsc.synthetic_region(n=max(n, 60), seed=seed)   # class means need all classes present
    c = c[c["parcel_id"] < n].copy(); ids = list(range(n))
    # only mineral-class parcels have own class in domain; peat parcels are fine too (4-class domain)
    p = nss.ScenarioProblem(ids, c, m, s)
    sol = sef.ExactSolver(p, nss.evaluate_batch, False)
    rng = np.random.default_rng(seed + 100)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p_edge]
    return co.CoupledProblem(sol, pairs), sol

def brute(cp, budget, lam):
    best, arg = -np.inf, None
    opts = [[-1] + cp.moves_of[p] for p in range(cp.n)]
    for combo in itertools.product(*opts):
        ch = np.array(combo)
        if not cp.feasible(ch, budget): continue
        J = cp.evaluate(ch, lam)["J"]
        if J > best + 1e-12: best, arg = J, ch
    return best, arg

# 1. lam = 0 reduces to the additive problem: coupled MILP == mckp exact
cp, sol = world(14, 1)
inst = mt.parcel_instance(sol)
for b in (0.2, 0.5):
    ch, info = cp.solve_exact(b, 0.0)
    _, g = mt.solve_exact(inst, b)
    check(f"lam=0 matches additive exact (budget {b})", abs(cp.evaluate(ch, 0)["carbon_t"] - g) <= 1e-6 * max(1, abs(g)))

# 2. brute force, several seeds / lambdas / budgets (tiny n)
for seed in (2, 3, 4):
    cp, sol = world(6, seed, p_edge=0.5)
    scale = np.abs(cp.mv_gain).mean()
    for lam_f in (0.0, 0.3, 1.0, 3.0):
        for b in (0.25, 0.6):
            lam = lam_f * scale
            bJ, _ = brute(cp, b, lam)
            ch, info = cp.solve_exact(b, lam, gap=1e-9)
            J = cp.evaluate(ch, lam)["J"]
            check(f"MILP == brute force (seed {seed}, lam {lam_f}x, budget {b})",
                  info["proven"] and abs(J - bJ) <= 1e-6 * max(1, abs(bJ)), f"MILP {J:.4f} brute {bJ:.4f}")
            check("  budget respected", cp.feasible(ch, b))
            for name, h in (("greedy_naive", cp.greedy_naive(b)), ("aware", cp.greedy_aware(b, lam)),
                            ("aware+LS", cp.local_search(cp.greedy_aware(b, lam), b, lam))):
                check(f"  {name} feasible and <= optimum", cp.feasible(h, b) and cp.evaluate(h, lam)["J"] <= bJ + 1e-6)

# 3. non-positive-gain "hole" parcels can be worth converting when lam is large (why they are kept)
found = False
for seed in range(10, 40):
    cp, sol = world(7, seed, p_edge=0.6)
    lam = 3 * np.abs(cp.mv_gain).mean()
    ch, _ = cp.solve_exact(0.9, lam, gap=1e-9)
    if any(cp.mv_gain[m] <= 0 for m in ch[ch >= 0]):
        found = True; break
check("optimum can include a non-positive-gain conversion (so they must not be dropped)", found)

# 4. cut / patches / evaluate against an independent reference
cp, sol = world(12, 5)
rng = np.random.default_rng(0)
ch = np.full(cp.n, -1)
for p in rng.choice(cp.n, 5, replace=False): ch[p] = rng.choice(cp.moves_of[p])
cls = cp.final_classes(ch)
ref = sum(cp.ew[e] for e in range(cp.ne) if cls[cp.ea[e]] != cls[cp.eb[e]])
check("boundary count matches loop reference", abs(cp.evaluate(ch, 1)["boundary"] - ref) < 1e-12)
chg = set(np.flatnonzero(ch >= 0)); seen, comps = set(), 0
adj = {p: [q for q, _ in cp.nbrs[p] if q in chg] for p in chg}
for p in chg:
    if p in seen: continue
    comps += 1; stack = [p]
    while stack:
        u = stack.pop()
        if u in seen: continue
        seen.add(u); stack += adj[u]
check("patch count matches DFS reference", cp.evaluate(ch, 1)["patches"] == comps)

# 5. NSGA-II: vectorised boundary == per-individual reference; carbon/disruption == evaluate_batch;
#    evaluator agrees with CoupledProblem.evaluate on the same assignments
try:
    prob, table = co.make_problem3(cp)
    X = np.stack([np.where(rng.random(cp.n) < 0.4, rng.integers(0, prob.xu + 1), prob._tables["baseline_idx_per_parcel"])
                  for _ in range(20)])
    out = {}; prob._evaluate(X, out)
    F = out["F"]
    ok = True
    for r in range(len(X)):
        choice = np.full(cp.n, -1)
        for p in range(cp.n):
            if X[r, p] != prob._tables["baseline_idx_per_parcel"][p]:
                choice[p] = [m for m in cp.moves_of[p] if cp.mv_k[m] == X[r, p]][0]
        ev = cp.evaluate(choice, 0)
        ok &= abs(F[r, 2] - ev["boundary"]) < 1e-9 and abs(-F[r, 0] - ev["carbon_t"]) < 1e-6 and abs(F[r, 1] - ev["disruption"]) < 1e-9
    check("NSGA-II 3-objective evaluator == CoupledProblem.evaluate (20 random individuals)", ok)
    seedsol = [cp.local_search(cp.greedy_aware(0.5, 0.0), 0.5, 0.0)]
    cells = [(0.3, 0.0), (0.5, 0.5 * np.abs(cp.mv_gain).mean())]
    Fn, Xn, pr = co.run_nsga2_3obj(cp, n_generations=5, pop_size=20, seed=1, seed_solutions=seedsol, cells=cells)
    check("NSGA-II 3-objective runs", Fn.shape[1] == 3 and len(Fn) > 0)
    ok = True
    for (b, lam), (J, x) in zip(cells, pr.archive):
        bi = pr._tables["baseline_idx_per_parcel"]; ch = np.full(cp.n, -1)
        for p in np.flatnonzero(x != bi): ch[p] = [m for m in cp.moves_of[p] if cp.mv_k[m] == x[p]][0]
        ev = cp.evaluate(ch, lam)
        ok &= abs(ev["J"] - J) < 1e-6 and cp.feasible(ch, b)
    check("archive entries re-evaluate to the stored J and respect the budget", ok)
    ev_seed = cp.evaluate(seedsol[0], cells[1][1])["J"]
    check("archive is at least as good as the seed it was given (cell with the seed's budget)", pr.archive[1][0] >= ev_seed - 1e-6)
except ImportError as e:
    print("SKIP NSGA-II checks:", e)

print("\n%d failure(s)" % len(fails)); sys.exit(1 if fails else 0)
