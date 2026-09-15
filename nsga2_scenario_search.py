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
from carbon_transition_delta import compute_branch_means, compute_branch_stds


class ScenarioProblem(Problem):
    """
    Each individual `X[i]` is a length-n_parcels array of class INDICES
    (not labels) for the masked region -- pymoo requires numeric arrays,
    so class<->index mapping happens at the Problem boundary and nowhere
    else, keeping score_scenario() itself untouched.
    """

    def __init__(self, parcel_ids, combined, branch_means, branch_stds,
                 class_labels):
        self.parcel_ids = parcel_ids
        self.combined = combined
        self.branch_means = branch_means
        self.branch_stds = branch_stds
        self.class_labels = class_labels  # index -> label
        self.label_to_idx = {c: i for i, c in enumerate(class_labels)}

        super().__init__(
            n_var=len(parcel_ids),
            n_obj=2,               # [-carbon_delta, disruption_fraction]
            n_ieq_constr=0,
            xl=0, xu=len(class_labels) - 1,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        f = np.zeros((X.shape[0], 2))
        for i, row in enumerate(X):
            new_classes = {
                pid: self.class_labels[int(cls_idx)]
                for pid, cls_idx in zip(self.parcel_ids, row)
            }
            score = score_scenario(
                self.parcel_ids, new_classes, self.combined,
                self.branch_means, self.branch_stds,
                # tillage/grassland args unused by the two objectives
                # below but required by score_scenario()'s signature
                tillage_yield_mean=0.0, tillage_yield_std=0.0,
                grassland_index_std=0.0,
            )
            f[i, 0] = -score["carbon_delta_vs_baseline_t"]  # negate: pymoo minimizes
            f[i, 1] = score["disruption_fraction"]
        out["F"] = f


class GNNSeededSampling(Sampling):
    """Initial population = real draws from the trained GNN, not random
    assignments -- see module docstring."""

    def __init__(self, parcel_ids, model, data, meta, idx_to_class):
        super().__init__()
        self.parcel_ids = parcel_ids
        self.model, self.data, self.meta, self.idx_to_class = model, data, meta, idx_to_class

    def _do(self, problem, n_samples, **kwargs):
        samples = sample_reallocation(
            self.parcel_ids, self.model, self.data, self.meta,
            self.idx_to_class, n_samples=n_samples, temperature=1.0,
        )
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for s_i, sample_id in enumerate(samples["sample_id"].unique()):
            this = samples[samples["sample_id"] == sample_id]
            cls_by_pid = dict(zip(this["parcel_id"], this["sampled_class"]))
            for p_i, pid in enumerate(self.parcel_ids):
                cls = cls_by_pid.get(pid, self.idx_to_class[0])
                X[s_i, p_i] = problem.label_to_idx[cls]
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
    """Each parcel independently has `prob` chance of being reset to a
    uniformly random class -- the categorical equivalent of Gaussian
    mutation for a continuous problem. `prob` should be small (roughly
    1/n_var is the standard default) so mutation nudges rather than
    destroys inherited structure."""

    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        prob = self.prob if self.prob is not None else 1.0 / problem.n_var
        mutate_mask = np.random.rand(*X.shape) < prob
        n_classes = problem.xu[0] + 1  # xu is the same upper bound for every parcel-variable
        random_classes = np.random.randint(0, n_classes, size=X.shape)
        return np.where(mutate_mask, random_classes, X)


def run_nsga2_search(parcel_ids, n_generations=50, pop_size=40):
    model, data, meta, idx_to_class = load_model_and_graph()
    combined = load_area_and_baseline()
    branch_means = compute_branch_means("data/parcel_yield_carbon.gpkg")
    branch_stds = compute_branch_stds("data/parcel_yield_carbon.gpkg")

    parcel_ids = filter_reallocatable(parcel_ids, combined)
    class_labels = sorted(set(idx_to_class.values()))

    problem = ScenarioProblem(parcel_ids, combined, branch_means, branch_stds, class_labels)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=GNNSeededSampling(parcel_ids, model, data, meta, idx_to_class),
        crossover=UniformParcelCrossover(),
        mutation=RandomResetMutation(),
        eliminate_duplicates=True,
    )

    print(f"Running NSGA-II: {pop_size} population x {n_generations} generations "
          f"over {len(parcel_ids)} parcels ({len(class_labels)} classes)...")
    result = minimize(problem, algorithm, ("n_gen", n_generations), seed=1, verbose=True)

    pareto_front = pd.DataFrame(result.F, columns=["neg_carbon_delta_t", "disruption_fraction"])
    pareto_front["carbon_delta_vs_baseline_t"] = -pareto_front["neg_carbon_delta_t"]
    pareto_front = pareto_front.drop(columns=["neg_carbon_delta_t"])
    pareto_front = pareto_front.sort_values("carbon_delta_vs_baseline_t", ascending=False)

    print(f"\nFinal Pareto front: {len(pareto_front)} non-dominated scenarios "
          f"(vs. sample-and-rank's typical ~2/10 non-dominated)")
    print(pareto_front.round(1).to_string(index=False))
    return result, pareto_front


if __name__ == "__main__":
    example_parcel_ids = list(range(100, 150))
    run_nsga2_search(example_parcel_ids, n_generations=50, pop_size=40)
