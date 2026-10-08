"""
Shared multiple-choice-knapsack (MCKP) tools for the robustness experiments.

An instance is a set of GROUPS (a parcel, or a block of parcels decided
together), each offering candidate MOVES (change to class c) with a weight
(hectares whose class changes -> counts as disruption) and a gain (t C). At most
one move per group; total weight <= budget x region area. Maximise total gain.

This is the same problem solve_exact_front.ExactSolver solves for single
parcels, written generically so blocks and perturbed gains can reuse it:
  * solve_exact  -- MILP (HiGHS), certified to 0.01% by default
  * solve_greedy -- density-greedy, the rule baseline_comparison.greedy_baseline
                    uses (best class per group by gain per ha; take groups in
                    descending density until the budget is reached), plus a
                    "fill" variant that keeps scanning after the first overshoot
test_experiments.py ties both to the existing, already-verified code.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from solve_exact_front import _silence_c_output

REALISTIC = ["tillage", "grassland", "forestry"]


@dataclass
class MCKP:
    group: np.ndarray       # (M,) group id of each move
    weight: np.ndarray      # (M,) hectares whose class changes
    gain: np.ndarray        # (M,) t C
    n_groups: int
    total_area: float       # hectares in the region; capacity = budget * total_area

    def with_gain(self, gain):
        return MCKP(self.group, self.weight, np.asarray(gain, dtype=float), self.n_groups, self.total_area)


def parcel_instance(solver):
    """One group per parcel, from an ExactSolver (same moves and gains it optimises)."""
    return MCKP(group=np.asarray(solver.vp), weight=np.asarray(solver.area)[solver.vp],
                gain=np.asarray(solver.gain, dtype=float), n_groups=solver.n, total_area=float(solver.total_area))


def solve_exact(inst, budget, gap=1e-4, time_limit=60):
    """Returns (indices of chosen moves, total gain). Moves with gain <= 0 are never
    chosen (doing nothing dominates them), so they are dropped before solving."""
    idx = np.flatnonzero(inst.gain > 0)
    if len(idx) == 0:
        return idx, 0.0
    g, w = inst.gain[idx], inst.weight[idx]
    _, inv = np.unique(inst.group[idx], return_inverse=True)
    A = csr_matrix((np.ones(len(idx)), (inv, np.arange(len(idx)))), shape=(inv.max() + 1, len(idx)))
    cap = budget * inst.total_area
    rhs = cap - 1e-6
    for _ in range(6):
        with _silence_c_output():
            res = milp(c=-g, constraints=[LinearConstraint(A, -np.inf, 1.0),
                                          LinearConstraint(csr_matrix(w[None, :]), -np.inf, max(rhs, 0.0))],
                       integrality=np.ones(len(idx)), bounds=Bounds(0, 1),
                       options={"mip_rel_gap": gap, "time_limit": time_limit})
        if res.x is None:
            raise RuntimeError(f"MILP found no solution (status {res.status}: {res.message})")
        chosen = idx[np.round(res.x).astype(int) == 1]
        used = float(inst.weight[chosen].sum())
        if used <= cap + 1e-9:
            return chosen, float(inst.gain[chosen].sum())
        rhs -= (used - cap) * 2 + 1e-6            # solver tolerance overshoot: tighten and retry
    raise RuntimeError("could not obtain a feasible MILP solution")


def solve_greedy(inst, budget, fill=False, tie_priority=None):
    """Density-greedy. fill=False stops at the first group that would overshoot (what
    baseline_comparison.greedy_baseline does); fill=True skips it and keeps scanning.
    Groups with exactly equal density are ordered by tie_priority (one number per group, lower
    first); the default is the group index, i.e. parcel_id order -- the rule greedy_baseline uses."""
    cap = budget * inst.total_area
    ok = np.flatnonzero((inst.gain > 0) & (inst.weight > 0))
    if len(ok) == 0:
        return ok, 0.0
    # Rounded so exactly-tied densities (common: baselines come from soil-association values)
    # stay tied despite float noise, then ties break by group index (= parcel_id order). This
    # is the SAME rule baseline_comparison.greedy_baseline uses.
    dens = np.round(inst.gain[ok] / inst.weight[ok], 9)
    o = np.lexsort((-dens, inst.group[ok]))           # per group: best density first
    ok, dens = ok[o], dens[o]
    first = np.r_[True, inst.group[ok][1:] != inst.group[ok][:-1]]
    best, bdens = ok[first], dens[first]               # one move per group
    tie = inst.group[best] if tie_priority is None else np.asarray(tie_priority)[inst.group[best]]
    order = np.lexsort((tie, -bdens))                            # primary: density desc; ties: tie priority
    chosen, used = [], 0.0
    for m in best[order]:
        if used + inst.weight[m] <= cap + 1e-12:
            chosen.append(m)
            used += inst.weight[m]
        elif not fill:
            break
    chosen = np.array(chosen, dtype=int)
    return chosen, float(inst.gain[chosen].sum()) if len(chosen) else 0.0


# ----------------------------------------------------------------- blocks
def class_gain_tables(solver):
    """Per-parcel gain (t C) and weight (ha) of converting the parcel to each realistic class
    (zero for its own class). Built from the ExactSolver's moves, i.e. the verified lookup table."""
    n = solver.n
    labels = solver.problem.per_parcel_labels
    own = (solver.problem.combined.set_index("parcel_id").loc[solver.problem.parcel_ids, "land_cover_class"].tolist())
    G = np.zeros((n, len(REALISTIC)))
    for m in range(solver.nv):
        lab = labels[solver.vp[m]][solver.vk[m]]
        if lab in REALISTIC:
            G[solver.vp[m], REALISTIC.index(lab)] = solver.gain[m]
    W = np.zeros_like(G)
    for p in range(n):
        for c, cls in enumerate(REALISTIC):
            if cls != own[p]:
                W[p, c] = solver.area[p]
    return G, W, own


def make_blocks(n, edge_pairs, block_size, rng):
    """Partition parcel indices 0..n-1 into spatially connected blocks of at most block_size
    (a proxy for a farm: neighbouring parcels decided together), by region growing over the
    adjacency graph from random seeds. edge_pairs: iterable of (i, j) index pairs."""
    nbrs = [[] for _ in range(n)]
    for i, j in edge_pairs:
        nbrs[i].append(j)
        nbrs[j].append(i)
    unassigned = np.ones(n, dtype=bool)
    blocks = []
    for seed in rng.permutation(n):
        if not unassigned[seed]:
            continue
        block, queue = [int(seed)], deque([int(seed)])
        unassigned[seed] = False
        while queue and len(block) < block_size:
            u = queue.popleft()
            cand = [v for v in nbrs[u] if unassigned[v]]
            rng.shuffle(cand)
            for v in cand:
                if len(block) >= block_size:
                    break
                unassigned[v] = False
                block.append(v)
                queue.append(v)
        blocks.append(np.array(block, dtype=int))
    return blocks


def block_instance(G, W, blocks, total_area):
    """Each block takes ONE class for all its parcels (a farm-level decision). Moves with
    no weight or no gain are dropped. Returns (MCKP, move_block, move_class)."""
    grp, wt, gn, mc = [], [], [], []
    for b, idx in enumerate(blocks):
        gb, wb = G[idx].sum(axis=0), W[idx].sum(axis=0)
        for c in range(G.shape[1]):
            if wb[c] > 0 and gb[c] > 0:
                grp.append(b); wt.append(wb[c]); gn.append(gb[c]); mc.append(c)
    inst = MCKP(np.array(grp, dtype=int), np.array(wt), np.array(gn), len(blocks), float(total_area))
    return inst, inst.group.copy(), np.array(mc, dtype=int)


def edges_to_index_pairs(edges_df, parcel_ids):
    """Restrict a graph edge table (columns a, b = parcel_id) to the candidates, as index pairs."""
    pos = {pid: i for i, pid in enumerate(parcel_ids)}
    keep = edges_df["a"].isin(pos) & edges_df["b"].isin(pos)
    return [(pos[a], pos[b]) for a, b in zip(edges_df.loc[keep, "a"], edges_df.loc[keep, "b"]) if a != b]
