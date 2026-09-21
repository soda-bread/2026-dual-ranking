"""Rank-and-crowding survival operators used by the primary methods."""

from __future__ import annotations

import numpy as np
from pymoo.core.survival import Survival
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


def _rank_and_crowd(
    pop,
    F_rank,
    F_crowd,
    n_survive,
    random_state,
    *,
    nds,
    crowding_func,
):
    """Select survivors without consuming RNG after the population is full."""

    F_rank = np.asarray(F_rank, dtype=float)
    F_crowd = np.asarray(F_crowd, dtype=float)
    if F_rank.ndim != 2 or F_crowd.ndim != 2:
        raise ValueError("ranking and crowding objectives must be 2D arrays.")
    if len(F_rank) != len(pop) or len(F_crowd) != len(pop):
        raise ValueError("ranking and crowding objectives must align with pop.")
    if not np.all(np.isfinite(F_rank)) or not np.all(np.isfinite(F_crowd)):
        raise ValueError("ranking and crowding objectives must be finite.")
    n_survive = len(pop) if n_survive is None else int(n_survive)
    if not 0 <= n_survive <= len(pop):
        raise ValueError("n_survive must be between zero and population size.")
    if n_survive == 0:
        return pop[[]]

    survivors = []
    fronts = nds.do(F_rank, n_stop_if_ranked=n_survive)
    for rank, front in enumerate(fronts):
        indices = np.arange(len(front))
        n_remove = max(0, len(survivors) + len(front) - n_survive)
        crowding = crowding_func.do(F_crowd[front, :], n_remove=n_remove)
        if n_remove:
            indices = randomized_argsort(
                crowding,
                order="descending",
                method="numpy",
                random_state=random_state,
            )[:-n_remove]

        for local_index, population_index in enumerate(front):
            pop[population_index].set("rank", rank)
            pop[population_index].set("crowding", crowding[local_index])
        survivors.extend(front[indices])
        if len(survivors) >= n_survive:
            break
    return pop[np.asarray(survivors[:n_survive], dtype=int)]


class Survival_standard(Survival):
    def __init__(self, nds=None, crowding_func="cd"):
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = get_crowding_function(crowding_func)

    def _do(self, problem, pop, *args, random_state=None, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        return _rank_and_crowd(
            pop,
            F,
            F,
            n_survive,
            random_state,
            nds=self.nds,
            crowding_func=self.crowding_func,
        )


class Survival_dr(Survival):
    """Original concatenated mean/upper-bound DR survival."""

    def __init__(
        self,
        nds=None,
        crowding_func="cd",
        alpha_f1=1,
        alpha_f2=1,
        alpha=None,
        alphas=None,
    ):
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = get_crowding_function(crowding_func)
        self.alpha_f1 = alpha_f1
        self.alpha_f2 = alpha_f2
        self.alpha = alpha
        self.alphas = None if alphas is None else np.asarray(alphas, dtype=float)

    def _do(self, problem, pop, *args, random_state=None, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        if self.alpha is not None:
            quantile_name = {0.8: "F_q80", 0.9: "F_q90", 0.95: "F_q95"}.get(
                self.alpha
            )
            if quantile_name is None:
                raise ValueError(
                    "alpha must be one of 0.8, 0.9, 0.95 for quantile DR."
                )
            F_upper = pop.get(quantile_name).astype(float, copy=False)
            if F_upper.shape != F.shape:
                raise ValueError("Upper-quantile predictions must match F.")
            if not np.all(np.isfinite(F_upper)):
                raise ValueError("Upper-quantile predictions must be finite.")
        else:
            F_std = pop.get("std").astype(float, copy=False)
            alphas = (
                self.alphas
                if self.alphas is not None
                else np.array([self.alpha_f1, self.alpha_f2], dtype=float)
            )
            if alphas.shape != (F.shape[1],):
                raise ValueError(
                    "Expected one DR alpha per objective; "
                    f"received {alphas.shape} for {F.shape[1]} objectives."
                )
            F_upper = F + alphas * F_std
        return _rank_and_crowd(
            pop,
            np.concatenate([F, F_upper], axis=1),
            F,
            n_survive,
            random_state,
            nds=self.nds,
            crowding_func=self.crowding_func,
        )


# Compatibility for notebooks and external scripts written before the public
# method-category name was shortened from ``dual-ranking`` to ``dr``.
Survival_dual_ranking = Survival_dr


class Survival_eb_shrinkage(Survival):
    """Three-regime empirical-Bayes dual-view survival (``ebu_dr``)."""

    def __init__(
        self,
        m,
        tau2,
        c,
        s2max,
        signal,
        *,
        view="concat",
        uninformative="sigma",
        nds=None,
        crowding_func="cd",
    ):
        super().__init__(filter_infeasible=True)
        self.m = np.asarray(m, dtype=float).reshape(-1)
        self.tau2 = np.asarray(tau2, dtype=float).reshape(-1)
        self.c = np.asarray(c, dtype=float).reshape(-1)
        self.s2max = np.asarray(s2max, dtype=float).reshape(-1)
        self.signal = np.asarray(signal, dtype=bool).reshape(-1)
        if not (
            self.m.shape
            == self.tau2.shape
            == self.c.shape
            == self.s2max.shape
            == self.signal.shape
        ):
            raise ValueError("EBU-DR parameters must have one value per objective.")
        if np.any(self.tau2 < 0) or np.any(self.c < 0) or np.any(self.s2max < 0):
            raise ValueError("tau2, c, and s2max must be non-negative.")
        if view != "concat":
            raise ValueError("The formal EBU-DR path supports view='concat' only.")
        if uninformative != "sigma":
            raise ValueError(
                "The formal EBU-DR path supports uninformative='sigma' only."
            )
        self.view = view
        self.uninformative = uninformative
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = get_crowding_function(crowding_func)

    def adjusted(self, F, std):
        F = np.asarray(F, dtype=float)
        std = np.asarray(std, dtype=float)
        if F.ndim != 2 or std.shape != F.shape:
            raise ValueError("F and std must be aligned 2D arrays.")
        if F.shape[1] != len(self.m):
            raise ValueError("EBU-DR parameters do not match objective count.")
        if not np.all(np.isfinite(F)) or not np.all(np.isfinite(std)):
            raise ValueError("F and std must be finite.")
        std = np.maximum(std, 0.0)
        s2 = np.minimum(std**2, self.s2max[None, :])
        denominator = self.tau2[None, :] + self.c[None, :] * s2
        shrinkage = np.ones_like(F)
        active = self.c[None, :] * s2 > 0
        np.divide(
            self.tau2[None, :],
            denominator,
            out=shrinkage,
            where=active,
        )
        adjusted = self.m[None, :] + shrinkage * (F - self.m[None, :])
        adjusted[:, ~self.signal] = std[:, ~self.signal]
        return adjusted

    def _do(self, problem, pop, *args, random_state=None, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        std = pop.get("std").astype(float, copy=False)
        adjusted = self.adjusted(F, std)
        return _rank_and_crowd(
            pop,
            np.concatenate([F, adjusted], axis=1),
            F,
            n_survive,
            random_state,
            nds=self.nds,
            crowding_func=self.crowding_func,
        )
