"""
NSGA-II multi-objective scenario search -- replaces generate_scenarios.py's
sample-and-rank approach with genuine evolutionary search over the
solution space, per Priority 3 (8 of 10 independent GNN draws were
dominated -- evidence the space needs exploring, not just sampling from).

REPRESENTATION: a candidate solution ("genome") is a dict
{parcel_id: class_label} over the masked region -- categorical, not
real-valued, so this uses pymoo's flexibility to plug in custom
sampling/crossover/mutation operators rather than its real-valued
defaults (which assume a continuous decision space).

SEEDING: the initial population is NOT random assignments -- it's
drawn from sample_reallocation() (the trained GNN), so evolution
starts from spatially-plausible, learned proposals and refines them
via crossover/mutation, rather than randomly exploring an enormous
unconstrained categorical space from scratch. This keeps the GNN's
contribution (contiguous, realistic patterns) while adding the
exploration pressure it currently lacks alone.

OBJECTIVES (all directly from score_scenario(), no new metric
invented): maximize carbon_delta_vs_baseline_t, minimize
disruption_fraction. pymoo minimizes by convention -> carbon
objective is negated internally.

NOT YET INCLUDED: tillage_yield / grassland_productivity_index as
explicit objectives -- they're only meaningful for the SUBSET of a
scenario in that class (per generate_scenarios.py's own documented
caveat), so folding them into a per-scenario objective needs a
decision on how to handle scenarios with ~0 tillage or grassland area
(objective is undefined / should it be 0? penalized?) -- flagged as a
design choice to make before adding them, not silently defaulted here.
"""

import numpy as np
import pandas as pd
import torch
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from sample_reallocation import load_model_and_graph, sample_reallocation
from generate_scenarios import (
    load_area_and_baseline, filter_reallocatable, score_scenario,
)


def build_lookup_tables(parcel_ids, combined, branch_means, branch_stds, per_parcel_labels):
    """One-time precomputation for vectorized batch evaluation -- see
    nsga2_vectorized_evaluate.py for the standalone version + correctness
    check against a reference implementation."""
    n_var = len(parcel_ids)
    max_domain = max(len(labels) for labels in per_parcel_labels)

    area_by_pid = dict(zip(combined["parcel_id"], combined["area_ha"]))
    carbon_by_pid = dict(zip(combined["parcel_id"], combined["carbon_t_c_per_ha_mean"]))
    baseline_class_by_pid = dict(zip(combined["parcel_id"], combined["land_cover_class"]))

    area_ha = np.array([area_by_pid[pid] for pid in parcel_ids])
    baseline_carbon_per_ha = np.array([carbon_by_pid[pid] for pid in parcel_ids])

    padded_carbon_lookup = np.zeros((n_var, max_domain))
    padded_std_lookup = np.zeros((n_var, max_domain))
    baseline_idx_per_parcel = np.full(n_var, -1, dtype=int)

    for p, pid in enumerate(parcel_ids):
        labels = per_parcel_labels[p]
        for idx, cls in enumerate(labels):
            padded_carbon_lookup[p, idx] = branch_means[cls]
            padded_std_lookup[p, idx] = branch_stds.get(cls, 0.0)
        baseline_cls = baseline_class_by_pid[pid]
        if baseline_cls in labels:
            baseline_idx_per_parcel[p] = labels.index(baseline_cls)
        # else stays -1 (sentinel): baseline class isn't a valid target for
        # this parcel (e.g. 'excluded') -- always counts as "changed",
        # matching score_scenario()'s `new_class != land_cover_class`.

    baseline_carbon_t = float((area_ha * baseline_carbon_per_ha).sum())
    total_area_ha = float(area_ha.sum())

    # Guard against NaN reaching pymoo -- a single NaN anywhere here
    # poisons EVERY individual's objective value (baseline_carbon_t is a
    # scalar summed across ALL parcels), which corrupts pymoo's internal
    # progress tracking 1-2 generations later with a confusing assertion
    # far from the actual cause (confirmed pattern: see pymoo
    # discussions#533 -- "Invalid progress" is caused by NaN in F).
    # Fail loudly HERE instead, pointing at the specific parcel_id.
    if np.isnan(baseline_carbon_per_ha).any():
        bad_pids = [pid for pid, c in zip(parcel_ids, baseline_carbon_per_ha) if np.isnan(c)]
        raise ValueError(
            f"NaN in carbon_t_c_per_ha_mean for parcel_id(s) {bad_pids} -- "
            "this will silently poison every individual's objective value "
            "if not fixed here. Check why the carbon model left these "
            "parcels unassigned (likely an 'excluded' or 'unresolved_crop' "
            "parcel that never got a value from finalize_carbon_estimate())."
        )
    if np.isnan(padded_carbon_lookup).any():
        raise ValueError(
            "NaN in branch_means for at least one class used in "
            "REALISTIC_TARGET_CLASSES or 'peatland' -- check branch_means "
            "dict for missing/NaN entries before running the search."
        )
    if total_area_ha <= 0:
        raise ValueError(
            f"total_area_ha is {total_area_ha} -- disruption_fraction would "
            "be NaN or Inf for every individual. Check for zero/invalid "
            "geometry area in the selected parcel_ids."
        )

    return {
        "area_ha": area_ha,
        "padded_carbon_lookup": padded_carbon_lookup,
        "padded_std_lookup": padded_std_lookup,
        "baseline_idx_per_parcel": baseline_idx_per_parcel,
        "baseline_carbon_t": baseline_carbon_t,
        "total_area_ha": total_area_ha,
    }


def evaluate_batch(X, tables):
    """Scores the ENTIRE population in one call via numpy fancy indexing,
    replacing a per-individual score_scenario() loop. X: (pop_size, n_var)
    int array of chosen class indices. Returns F: (pop_size, 2) =
    [-carbon_delta_t, disruption_fraction]."""
    area_ha = tables["area_ha"]
    col_idx = np.arange(area_ha.shape[0])

    new_carbon_per_ha = tables["padded_carbon_lookup"][col_idx, X]
    total_carbon_t = (area_ha[None, :] * new_carbon_per_ha).sum(axis=1)
    carbon_delta_t = total_carbon_t - tables["baseline_carbon_t"]

    changed = X != tables["baseline_idx_per_parcel"][None, :]
    disrupted_area = (changed * area_ha[None, :]).sum(axis=1)
    disruption_fraction = disrupted_area / tables["total_area_ha"]

    return np.column_stack([-carbon_delta_t, disruption_fraction])


class ScenarioProblem(Problem):
    """
    Each individual `X[i]` is a length-n_parcels array of class INDICES
    for the masked region. Indices are now PER-PARCEL, not global: a
    peat parcel's valid domain is [tillage, grassland, forestry,
    peatland] (4 options); every other parcel's is [tillage, grassland,
    forestry] (3 options, NO peatland -- a mineral-soil parcel cannot
    become peat by reallocation). `excluded` and `unresolved_crop` are
    never valid targets for anyone -- they're data-artifact/infrastructure
    buckets, not real land-use outcomes.

    pymoo supports per-variable bounds (xl/xu as arrays, not scalars),
    which is what makes this heterogeneous domain size possible without
    a custom Problem subclass beyond this.
    """

    def __init__(self, parcel_ids, combined, branch_means, branch_stds):
        self.parcel_ids = parcel_ids
        self.combined = combined
        self.branch_means = branch_means
        self.branch_stds = branch_stds

        # Per-parcel valid label list + per-parcel xu (= len - 1)
        original_class = dict(zip(combined["parcel_id"], combined["land_cover_class"]))
        self.per_parcel_labels = []
        xu = []
        for pid in parcel_ids:
            is_peat = original_class.get(pid) == "peatland"
            labels = REALISTIC_TARGET_CLASSES + (["peatland"] if is_peat else [])
            self.per_parcel_labels.append(labels)
            xu.append(len(labels) - 1)
        self.label_to_idx = [
            {c: i for i, c in enumerate(labels)} for labels in self.per_parcel_labels
        ]

        # Vectorized evaluation tables -- precomputed ONCE here, not per
        # evaluation. Benchmarking showed ~90ms/evaluation with the old
        # per-individual score_scenario() loop at 444 parcels (DataFrame
        # construction + groupby overhead, not raw compute); this batches
        # the entire population through numpy fancy-indexing instead.
        # Verified numerically identical to a reference implementation --
        # see nsga2_vectorized_evaluate.py's standalone correctness check.
        self._tables = build_lookup_tables(
            parcel_ids, combined, branch_means, branch_stds, self.per_parcel_labels
        )

        super().__init__(
            n_var=len(parcel_ids), n_obj=2, n_ieq_constr=0,
            xl=0, xu=np.array(xu, dtype=int), vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = evaluate_batch(X, self._tables)


REALISTIC_TARGET_CLASSES = ["tillage", "grassland", "forestry"]


class GNNSeededSampling(Sampling):
    """Initial population = real draws from the trained GNN, not random
    assignments -- see module docstring. FALLBACK: if the GNN proposes a
    class outside a parcel's valid domain (e.g. `peatland` for a
    non-peat parcel, or `excluded`/`unresolved_crop` for anyone), that
    parcel keeps its ORIGINAL class in this individual instead --
    silently dropping to index 0 would inject a specific, misleading
    bias (always defaulting to `tillage`); keeping the original class
    is a neutral fallback that just means "the GNN's suggestion here
    wasn't usable, don't change this parcel in this seed individual"
    rather than fabricating a preference."""

    def __init__(self, parcel_ids, model, data, meta, idx_to_class):
        super().__init__()
        self.parcel_ids = parcel_ids
        self.model, self.data, self.meta, self.idx_to_class = model, data, meta, idx_to_class

    def _do(self, problem, n_samples, **kwargs):
        samples = sample_reallocation(
            self.parcel_ids, self.model, self.data, self.meta,
            self.idx_to_class, n_samples=n_samples, temperature=1.0,
        )
        original_class = dict(zip(problem.combined["parcel_id"], problem.combined["land_cover_class"]))
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        n_fallback = 0
        for s_i, sample_id in enumerate(samples["sample_id"].unique()):
            this = samples[samples["sample_id"] == sample_id]
            cls_by_pid = dict(zip(this["parcel_id"], this["sampled_class"]))
            for p_i, pid in enumerate(self.parcel_ids):
                proposed = cls_by_pid.get(pid)
                valid_labels = problem.per_parcel_labels[p_i]
                if proposed in valid_labels:
                    X[s_i, p_i] = problem.label_to_idx[p_i][proposed]
                else:
                    n_fallback += 1
                    fallback_cls = original_class.get(pid, valid_labels[0])
                    # original class itself might not be in valid_labels
                    # (e.g. an 'excluded' or 'unresolved_crop' parcel's
                    # own original state) -- last-resort default to the
                    # domain's first entry in that edge case
                    X[s_i, p_i] = problem.label_to_idx[p_i].get(fallback_cls, 0)
        if n_fallback:
            print(f"GNN-seeded sampling: {n_fallback} proposals fell outside "
                  f"the parcel's valid target domain (out of "
                  f"{n_samples * problem.n_var} total) -- fell back to "
                  "original class for those.")
        return X


class UniformParcelCrossover(Crossover):
    """Per-parcel uniform crossover: each parcel independently inherits
    its class from parent A or B with 50% probability. Simple, and a
    defensible default for a spatial problem where NO fixed 1D ordering
    of parcels is meaningful (unlike e.g. a tour in TSP) -- a
    block/spatial crossover (contiguous sub-regions swapped) would be a
    natural upgrade if plain uniform crossover under-preserves spatial
    contiguity in practice; worth checking empirically before adding
    that complexity."""

    def __init__(self):
        super().__init__(2, 2)  # 2 parents in, 2 children out

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        offspring = np.empty_like(X)
        for k in range(n_matings):
            mask = np.random.rand(n_var) < 0.5
            offspring[0, k, :] = np.where(mask, X[0, k, :], X[1, k, :])
            offspring[1, k, :] = np.where(mask, X[1, k, :], X[0, k, :])
        return offspring


class RandomResetMutation(Mutation):
    """Each parcel independently has `gene_prob` chance of being reset to
    a uniformly random class -- the categorical equivalent of Gaussian
    mutation for a continuous problem. `gene_prob` should be small
    (roughly 1/n_var is the standard default) so mutation nudges rather
    than destroys inherited structure.

    NOTE: deliberately named `gene_prob`, not `prob` -- pymoo's Mutation
    base class manages `self.prob` itself internally (per-INDIVIDUAL
    probability of entering mutation at all, wrapped in its own
    Parameter machinery), so overwriting it directly breaks that
    machinery. Here `prob=1.0` is passed to the base class (every
    individual always enters _do()), and the actual per-PARCEL
    mutation rate is handled entirely inside _do() via gene_prob.
    """

    def __init__(self, gene_prob=None):
        super().__init__(prob=1.0)
        self.gene_prob = gene_prob

    def _do(self, problem, X, **kwargs):
        gene_prob = self.gene_prob if self.gene_prob is not None else 1.0 / problem.n_var
        mutate_mask = np.random.rand(*X.shape) < gene_prob
        # xu is now PER-VARIABLE (peat parcels: 3, others: 2) -- draw each
        # column's random replacement against ITS OWN bound, not a single
        # shared one (problem.xu[0] would be wrong for every other column
        # now that domains differ in size per parcel).
        n_classes_per_col = problem.xu + 1  # shape (n_var,)
        random_classes = np.random.randint(0, n_classes_per_col, size=X.shape)
        return np.where(mutate_mask, random_classes, X)


def run_nsga2_search(parcel_ids, n_generations=200, pop_size=200, seed=1):
    # pymoo's seed= only controls numpy's global RNG (crossover/mutation
    # randomness). GNNSeededSampling draws via torch.multinomial, which
    # uses torch's SEPARATE random state -- without also seeding torch,
    # different pymoo seeds would still start from a nearly-identical
    # initial population, undermining what a multi-seed robustness check
    # is meant to show (that the RESULT is robust, not just the search
    # dynamics on top of one fixed starting point).
    torch.manual_seed(seed)

    model, data, meta, idx_to_class = load_model_and_graph()
    combined = load_area_and_baseline()
    # Computed directly from the already-loaded frame -- avoids re-reading
    # the full geometry-bearing GPKG from disk twice more (see
    # baseline_comparison.py's branch_means_from_loaded() for the same fix
    # applied there).
    branch_means = combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].mean().to_dict()
    branch_stds = combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].std().to_dict()

    parcel_ids = filter_reallocatable(parcel_ids, combined)

    # Drop parcels with no assigned carbon baseline at all (NaN, not even
    # the 0.0 'excluded' land should get) -- same reasoning as
    # IMMUTABLE_CLASSES exclusion: a parcel with no defensible baseline
    # isn't a valid candidate for THIS analysis, whatever upstream data
    # gap caused it. Fixing why these parcels never got a value from
    # finalize_carbon_estimate() is a separate, real question worth
    # investigating in carbon_model_v1.py -- this is a scoping decision
    # for the search, not a fix for the underlying gap.
    carbon_by_pid = dict(zip(combined["parcel_id"], combined["carbon_t_c_per_ha_mean"]))
    nan_pids = [pid for pid in parcel_ids if pd.isna(carbon_by_pid.get(pid))]
    if nan_pids:
        print(f"Excluded {len(nan_pids)} parcel(s) with no assigned carbon "
              f"baseline from the reallocation target set: {nan_pids} -- "
              "these remain as fixed context, not candidates, pending "
              "investigation of why finalize_carbon_estimate() left them "
              "unassigned.")
        parcel_ids = [pid for pid in parcel_ids if pid not in set(nan_pids)]

    # Pre-filter to just the region's parcels ONCE, here -- score_scenario()
    # internally does combined[combined["parcel_id"].isin(parcel_ids)],
    # which is cheap against a ~40-row frame but expensive run 2,000+ times
    # (pop_size * n_generations) against the FULL national table (~1.5M
    # rows, per the branch-mean counts). Same filter, same result, just
    # done once instead of on every fitness evaluation.
    region_combined = combined[combined["parcel_id"].isin(parcel_ids)].copy()

    problem = ScenarioProblem(parcel_ids, region_combined, branch_means, branch_stds)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=GNNSeededSampling(parcel_ids, model, data, meta, idx_to_class),
        crossover=UniformParcelCrossover(),
        mutation=RandomResetMutation(),
        eliminate_duplicates=True,
    )

    n_peat = sum(len(labels) == 4 for labels in problem.per_parcel_labels)
    print(f"Running NSGA-II: {pop_size} population x {n_generations} generations "
          f"over {len(parcel_ids)} parcels "
          f"({n_peat} peat parcels with 4-class domain, "
          f"{len(parcel_ids) - n_peat} others with 3-class domain)...")
    result = minimize(problem, algorithm, ("n_gen", n_generations), seed=seed, verbose=True)

    pareto_front = pd.DataFrame(result.F, columns=["neg_carbon_delta_t", "disruption_fraction"])
    pareto_front["carbon_delta_vs_baseline_t"] = -pareto_front["neg_carbon_delta_t"]
    pareto_front = pareto_front.drop(columns=["neg_carbon_delta_t"])
    pareto_front = pareto_front.sort_values("carbon_delta_vs_baseline_t", ascending=False)

    print(f"\nFinal Pareto front: {len(pareto_front)} non-dominated scenarios "
          f"(vs. sample-and-rank's typical ~2/10 non-dominated)")
    print(pareto_front.round(1).to_string(index=False))
    return result, pareto_front


def load_saved_subregion_parcel_ids(path="data/offaly_subregion_parcel_ids.txt"):
    """Loads the real, verified case-study region -- NOT a placeholder
    range. Use this everywhere a script needs parcel_ids for the actual
    case study, rather than a hardcoded example range."""
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


if __name__ == "__main__":
    parcel_ids = load_saved_subregion_parcel_ids()
    print(f"Loaded {len(parcel_ids)} parcel_ids from the saved case-study region")

    # Multi-seed robustness check: does the NSGA-II advantage over
    # greedy hold up across independent runs, or was seed=1 a lucky
    # draw? Each seed reseeds BOTH pymoo's numpy RNG (crossover/
    # mutation) and torch (GNN-seeded initial population) -- see the
    # comment in run_nsga2_search() for why both are needed.
    all_fronts = {}
    for seed in [1, 2, 3, 4]:
        print(f"\n{'='*20} SEED {seed} {'='*20}")
        result, pareto_front = run_nsga2_search(parcel_ids, n_generations=200, pop_size=200, seed=seed)
        all_fronts[seed] = pareto_front

    print(f"\n{'='*20} SUMMARY ACROSS {len(all_fronts)} SEEDS {'='*20}")
    for seed, front in all_fronts.items():
        best = front["carbon_delta_vs_baseline_t"].max()
        best_row = front.loc[front["carbon_delta_vs_baseline_t"].idxmax()]
        print(f"seed={seed}: best={best:,.1f} t C at disruption={best_row['disruption_fraction']:.1f}, "
              f"front spans disruption {front['disruption_fraction'].min():.1f}-{front['disruption_fraction'].max():.1f}, "
              f"n_nondominated={len(front)}")
