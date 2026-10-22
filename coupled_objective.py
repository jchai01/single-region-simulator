"""
A COUPLED objective: carbon gain minus a boundary (fragmentation) penalty.

    maximise   J = carbon_gain(t C)  -  lam * (increase in the number of neighbouring parcel pairs
                                               that end up in DIFFERENT land-use classes)
    (J is measured against doing nothing, so J >= 0 at the optimum and relative gaps are meaningful.)
    subject to  sum of changed hectares <= budget x region area,  one class per parcel.

With lam = 0 this is exactly the additive knapsack the paper solves, where fill-greedy is
essentially optimal. With lam > 0 a parcel's value depends on its neighbours' final classes, so
the problem is no longer a knapsack and density-greedy has no guarantee. This module provides:

  * CoupledProblem          -- built from an ExactSolver (forced_parcels_must_change=False) + edge pairs
  * evaluate()              -- carbon, disruption, boundary count, objective J, patch count
  * solve_exact()           -- MILP with one boundary variable per edge (HiGHS); reports the achieved gap.
                               Its LP relaxation is weak, so the time limit is a parameter and a
                               solve that stops at the limit says so (gap, dual bound) -- never
                               present such a result as a proven optimum.
  * greedy_naive()          -- the paper's fill-greedy, blind to the boundary term
  * greedy_aware()          -- greedy on (gain - lam * change in boundary) / hectares, recomputed
  * local_search()          -- single-parcel improving moves (add / drop / switch), exact objective
  * ScenarioProblem3 + run_nsga2_3obj() -- NSGA-II with boundary count as a third objective

Moves with gain <= 0 are KEPT here: converting a "hole" parcel can pay for itself through the
boundary term, so mckp_tools.solve_exact's "drop non-positive gains" shortcut would be wrong.
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from solve_exact_front import _silence_c_output
import mckp_tools as mt


class CoupledProblem:
    def __init__(self, solver, edge_pairs, edge_weights=None):
        if solver.forced_must_change or solver.sentinel.any():
            raise ValueError("coupled study needs an ExactSolver built with forced_parcels_must_change=False "
                             "and no parcels whose own class is outside its domain")
        self.solver = solver
        self.n = solver.n
        self.area = solver.area
        self.total_area = solver.total_area
        labels, problem = solver.problem.per_parcel_labels, solver.problem
        own = problem.combined.set_index("parcel_id").loc[problem.parcel_ids, "land_cover_class"].tolist()
        names = sorted({l for ls in labels for l in ls} | set(own))
        self.class_names = names
        cid = {c: i for i, c in enumerate(names)}
        self.base_cls = np.array([cid[c] for c in own])
        self.mv_parcel = np.asarray(solver.vp)
        self.mv_k = np.asarray(solver.vk)
        self.mv_gain = np.asarray(solver.gain, dtype=float)
        self.mv_cls = np.array([cid[labels[p][k]] for p, k in zip(self.mv_parcel, self.mv_k)])
        self.mv_weight = self.area[self.mv_parcel]
        self.nm = len(self.mv_parcel)
        self.moves_of = [[] for _ in range(self.n)]
        for m, p in enumerate(self.mv_parcel):
            self.moves_of[p].append(m)

        pairs = [(int(i), int(j)) for i, j in edge_pairs if i != j]
        uniq = {}
        for k, (i, j) in enumerate(pairs):
            key = (min(i, j), max(i, j))
            uniq[key] = uniq.get(key, 0.0) + (1.0 if edge_weights is None else float(edge_weights[k]))
        self.ea = np.array([k[0] for k in uniq], dtype=int)
        self.eb = np.array([k[1] for k in uniq], dtype=int)
        self.ew = np.array(list(uniq.values()), dtype=float)
        self.ne = len(self.ea)
        self.nbrs = [[] for _ in range(self.n)]
        for a, b, w in zip(self.ea, self.eb, self.ew):
            self.nbrs[a].append((b, w))
            self.nbrs[b].append((a, w))
        self.base_cut = float(self.ew[self.base_cls[self.ea] != self.base_cls[self.eb]].sum())

    # ---------------------------------------------------------------- evaluation
    def choice_from_moves(self, moves):
        choice = np.full(self.n, -1, dtype=int)
        for m in np.asarray(moves, dtype=int):
            assert choice[self.mv_parcel[m]] == -1, "two moves chosen for one parcel"
            choice[self.mv_parcel[m]] = m
        return choice

    def final_classes(self, choice):
        cls = self.base_cls.copy()
        ch = choice >= 0
        cls[ch] = self.mv_cls[choice[ch]]
        return cls

    def cut(self, cls):
        return float(self.ew[cls[self.ea] != cls[self.eb]].sum())

    def n_patches(self, choice):
        """Connected groups of changed parcels (adjacent changed parcels form one patch)."""
        ch = choice >= 0
        if not ch.any():
            return 0
        keep = ch[self.ea] & ch[self.eb]
        g = coo_matrix((np.ones(keep.sum()), (self.ea[keep], self.eb[keep])), shape=(self.n, self.n))
        return int(connected_components(g, directed=False)[0] - (~ch).sum())

    def evaluate(self, choice, lam):
        ch = choice >= 0
        carbon = float(self.mv_gain[choice[ch]].sum())
        disruption = float(self.area[ch].sum() / self.total_area)
        cut = self.cut(self.final_classes(choice))
        return {"carbon_t": carbon, "disruption": disruption, "boundary": cut,
                "boundary_vs_baseline": cut - self.base_cut, "J": carbon - lam * (cut - self.base_cut),
                "changed_parcels": int(ch.sum()), "patches": self.n_patches(choice)}

    def feasible(self, choice, budget):
        return float(self.area[choice >= 0].sum()) <= budget * self.total_area + 1e-9

    # ---------------------------------------------------------------- exact MILP
    def solve_exact(self, budget, lam, gap=1e-4, time_limit=120):
        """Returns (choice, info). info has status, achieved relative gap, dual bound (an upper
        bound on J, in t C, up to the constant -lam*0: J = carbon - lam*boundary) and wall time."""
        n, nm, ne = self.n, self.nm, self.ne
        nvar = nm + ne                                     # z_m (moves) then y_e (edge cut)
        cvec = np.r_[-self.mv_gain, lam * self.ew]
        rows, cols, vals, rhs = [], [], [], []
        r = 0
        # one move per parcel
        for p in range(n):
            for m in self.moves_of[p]:
                rows.append(r); cols.append(m); vals.append(1.0)
            rhs.append(1.0); r += 1
        A1 = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
        # boundary: y_e >= s_ic - s_jc and >= s_jc - s_ic for every class c
        # s_pc = [base_p == c] - sum(z of p's moves)   if c == base_p
        #      = z of p's move into c (0 if none)       otherwise
        rows, cols, vals, lb = [], [], [], []
        r2 = 0
        move_to = [{self.mv_cls[m]: m for m in self.moves_of[p]} for p in range(n)]

        def s_terms(p, c):
            if c == self.base_cls[p]:
                return 1.0, [(m, -1.0) for m in self.moves_of[p]]
            m = move_to[p].get(c)
            return 0.0, ([(m, 1.0)] if m is not None else [])

        for e in range(ne):
            i, j = self.ea[e], self.eb[e]
            for c in {self.base_cls[i], self.base_cls[j], *move_to[i], *move_to[j]}:
                ai, ti = s_terms(i, c)
                aj, tj = s_terms(j, c)
                for sgn in (1.0, -1.0):                    # y_e + sgn*(s_jc - s_ic) >= 0
                    rows.append(r2); cols.append(nm + e); vals.append(1.0)
                    for m, v in ti:
                        rows.append(r2); cols.append(m); vals.append(-sgn * v)
                    for m, v in tj:
                        rows.append(r2); cols.append(m); vals.append(sgn * v)
                    lb.append(sgn * (ai - aj))             # y_e + sgn*(tj - ti) >= sgn*(ai - aj)
                    r2 += 1
        A2 = coo_matrix((vals, (rows, cols)), shape=(r2, nvar)).tocsr()
        area_row = csr_matrix(np.r_[self.mv_weight, np.zeros(ne)][None, :])
        cap = budget * self.total_area
        rhs_area = cap - 1e-6
        t0 = time.time()
        for _ in range(6):
            with _silence_c_output():
                res = milp(c=cvec,
                           constraints=[LinearConstraint(A1, -np.inf, 1.0),
                                        LinearConstraint(A2, np.array(lb), np.inf),
                                        LinearConstraint(area_row, -np.inf, max(rhs_area, 0.0))],
                           integrality=np.r_[np.ones(nm), np.zeros(ne)],   # y_e is forced 0/1 by z
                           bounds=Bounds(np.zeros(nvar), np.ones(nvar)),
                           options={"mip_rel_gap": gap, "time_limit": time_limit})
            if res.x is None:
                raise RuntimeError(f"coupled MILP found no solution (status {res.status}: {res.message})")
            z = np.round(res.x[:nm]).astype(int)
            moves = np.flatnonzero(z == 1)
            used = float(self.mv_weight[moves].sum())
            if used <= cap + 1e-9:
                break
            rhs_area -= (used - cap) * 2 + 1e-6
        else:
            raise RuntimeError("could not obtain a feasible coupled MILP solution")
        choice = self.choice_from_moves(moves)
        J = self.evaluate(choice, lam)["J"]
        dual = -float(getattr(res, "mip_dual_bound", np.nan)) + lam * self.base_cut   # milp minimises -(carbon - lam*cut)
        info = {"status": int(res.status), "proven": bool(res.success),
                "gap": float(getattr(res, "mip_gap", np.nan) or 0.0), "dual_bound_J": dual,
                "seconds": time.time() - t0, "J": J}
        return choice, info

    # ---------------------------------------------------------------- heuristics
    def greedy_naive(self, budget):
        """The paper's fill-greedy: ranks by carbon per hectare, ignores the boundary term."""
        inst = mt.parcel_instance(self.solver)
        moves, _ = mt.solve_greedy(inst, budget, fill=True)
        return self.choice_from_moves(moves)

    def _delta_cut(self, p, newc, cls):
        cur = cls[p]
        d = 0.0
        for q, w in self.nbrs[p]:
            d += w * (float(newc != cls[q]) - float(cur != cls[q]))
        return d

    def greedy_aware(self, budget, lam):
        """Repeatedly take the move with the best (gain - lam * change in boundary) per hectare
        among those that still fit and have positive net value. Marginals are refreshed for the
        moved parcel's neighbours after every pick."""
        cap = budget * self.total_area
        choice = np.full(self.n, -1, dtype=int)
        cls = self.base_cls.copy()
        net = self.mv_gain - lam * np.array([self._delta_cut(self.mv_parcel[m], self.mv_cls[m], cls)
                                             for m in range(self.nm)])
        alive = np.ones(self.nm, dtype=bool)
        used = 0.0
        while True:
            fits = alive & (used + self.mv_weight <= cap + 1e-12) & (net > 1e-12)
            if not fits.any():
                break
            score = np.where(fits, net / self.mv_weight, -np.inf)
            m = int(np.argmax(score))
            p = self.mv_parcel[m]
            choice[p] = m
            cls[p] = self.mv_cls[m]
            used += self.mv_weight[m]
            alive[self.moves_of[p]] = False
            for q, _ in self.nbrs[p]:
                if choice[q] == -1:
                    for mm in self.moves_of[q]:
                        net[mm] = self.mv_gain[mm] - lam * self._delta_cut(q, self.mv_cls[mm], cls)
        return choice

    def local_search(self, choice, budget, lam, max_passes=50):
        """Best-improvement over single-parcel changes (unchanged / any move), exact objective,
        budget respected. Never worsens J."""
        cap = budget * self.total_area
        choice = choice.copy()
        cls = self.final_classes(choice)
        used = float(self.area[choice >= 0].sum())
        for _ in range(max_passes):
            improved = False
            for p in range(self.n):
                cur_m = choice[p]
                cur_gain = self.mv_gain[cur_m] if cur_m >= 0 else 0.0
                cur_w = self.area[p] if cur_m >= 0 else 0.0
                best_dJ, best_opt = 1e-9, None
                for opt in [-1] + self.moves_of[p]:
                    if opt == cur_m:
                        continue
                    w = self.area[p] if opt >= 0 else 0.0
                    if used - cur_w + w > cap + 1e-12:
                        continue
                    newc = self.mv_cls[opt] if opt >= 0 else self.base_cls[p]
                    g = self.mv_gain[opt] if opt >= 0 else 0.0
                    dJ = (g - cur_gain) - lam * self._delta_cut(p, newc, cls)
                    if dJ > best_dJ:
                        best_dJ, best_opt = dJ, opt
                if best_opt is not None:
                    used += (self.area[p] if best_opt >= 0 else 0.0) - cur_w
                    choice[p] = best_opt
                    cls[p] = self.mv_cls[best_opt] if best_opt >= 0 else self.base_cls[p]
                    improved = True
            if not improved:
                break
        return choice


# ============================================================ NSGA-II with a boundary objective
def class_id_table(problem_cp):
    """(n, max_domain) final-class id for each parcel's search index X[p]; -1 in padding."""
    sol = problem_cp.solver
    labels = sol.problem.per_parcel_labels
    cid = {c: i for i, c in enumerate(problem_cp.class_names)}
    T = -np.ones((problem_cp.n, sol.lookup.shape[1]), dtype=int)
    for p, ls in enumerate(labels):
        for k, l in enumerate(ls):
            T[p, k] = cid[l]
    return T


def boundary_batch(X, table, ea, eb, ew):
    """Boundary count of every individual in a population X (pop, n) in one numpy pass."""
    cls = table[np.arange(X.shape[1])[None, :], X]
    return ((cls[:, ea] != cls[:, eb]) * ew[None, :]).sum(axis=1)


def make_problem3(problem_cp, cells=()):
    """A 3-objective pymoo problem: [-carbon, disruption, boundary]. Imported lazily so this
    module does not need pymoo unless the NSGA-II comparison is run.

    cells = [(budget, lam), ...]: while it searches, the problem keeps an ARCHIVE of the best J seen
    for each cell among EVERY individual it ever evaluated (disruption <= budget). NSGA-II's final
    population is truncated by crowding distance over the whole 3-D front, which can discard the
    best solution for one particular (budget, lam) cell, even a seeded greedy solution; the archive
    is immune to that, so it is the fair measure of what the search found."""
    import nsga2_scenario_search as nss
    base = problem_cp.solver.problem
    table = class_id_table(problem_cp)
    ea, eb, ew = problem_cp.ea, problem_cp.eb, problem_cp.ew

    class ScenarioProblem3(nss.ScenarioProblem):
        def __init__(self):
            super().__init__(base.parcel_ids, base.combined, base.branch_means, base.branch_stds)
            self.n_obj = 3
            self.cells = list(cells)
            self.archive = [(-np.inf, None)] * len(self.cells)

        def _evaluate(self, X, out, *args, **kwargs):
            F2 = nss.evaluate_batch(X, self._tables)
            F = np.column_stack([F2, boundary_batch(X, table, ea, eb, ew)])
            out["F"] = F
            for c, (b, lam) in enumerate(self.cells):
                J = np.where(F[:, 1] <= b + 1e-9, -F[:, 0] - lam * (F[:, 2] - problem_cp.base_cut), -np.inf)
                r = int(np.argmax(J))
                if J[r] > self.archive[c][0]:
                    self.archive[c] = (float(J[r]), X[r].copy())

    return ScenarioProblem3(), table


def run_nsga2_3obj(problem_cp, n_generations=200, pop_size=200, seed=1, seed_solutions=(), cells=()):
    """NSGA-II on [-carbon, disruption, boundary] with the existing operators. Initial population:
    random-domain draws at random disruption levels (no GNN needed), with `seed_solutions` (a list
    of `choice` vectors, e.g. greedy solutions) placed in the first slots when given, so the
    comparison is not against a handicapped start. Returns F, X, problem for the final set;
    problem.archive[c] = (best J, X) for each requested cell."""
    import nsga2_scenario_search as nss
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.sampling import Sampling
    from pymoo.optimize import minimize

    prob, _ = make_problem3(problem_cp, cells)
    base_idx = np.asarray(prob._tables["baseline_idx_per_parcel"])
    rng = np.random.default_rng(seed)
    if len(seed_solutions) > pop_size:
        raise ValueError(f"{len(seed_solutions)} seed solutions do not fit in a population of {pop_size}")
    seeds = []
    for ch in seed_solutions:
        x = base_idx.copy()
        idx = np.flatnonzero(ch >= 0)
        x[idx] = problem_cp.mv_k[ch[idx]]
        seeds.append(x)

    class Init(Sampling):
        def _do(self, problem, n_samples, **kw):
            X = np.tile(base_idx, (n_samples, 1))
            for i in range(n_samples):
                frac = rng.uniform(0.0, 0.7)
                flip = rng.random(problem.n_var) < frac
                X[i, flip] = rng.integers(0, problem.xu[flip] + 1)
            for i, s in enumerate(seeds[:n_samples]):
                X[i] = s
            return X

    alg = NSGA2(pop_size=pop_size, sampling=Init(), crossover=nss.UniformParcelCrossover(),
                mutation=nss.RandomResetMutation(), eliminate_duplicates=True)
    res = minimize(prob, alg, ("n_gen", n_generations), seed=seed, verbose=False)
    return res.F, res.X, prob
