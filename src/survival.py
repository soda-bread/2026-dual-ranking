import warnings

import numpy as np
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function


class Survival_standard(Survival):
    def __init__(self, nds=None, crowding_func='cd'):
        crowding_func_ = get_crowding_function(crowding_func)
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_

    def _do(self, problem, pop, *args, random_state=None, n_survive=None, **kwargs):
        F = pop.get('F').astype(float, copy=False)
        survivors = []
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            I = np.arange(len(front))
            if len(survivors) + len(I) > n_survive:
                n_remove = len(survivors) + len(front) - n_survive
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=n_remove)
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy', random_state=random_state)
                I = I[:-n_remove]
            else:
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=0)

            for j, i in enumerate(front):
                pop[i].set('rank', k)
                pop[i].set('crowding', crowding_of_front[j])
            survivors.extend(front[I])
        return pop[survivors]


def find_upper_alpha(
    model,
    X_val,
    y_val,
    target_coverage=0.90,
    alpha_max=500.0,
    alpha_step=0.01,
):
    """Find the first mean + alpha * std upper bound with target coverage."""
    mean, std = model.predict(X_val)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    if mean.shape != y_val.shape or std.shape != y_val.shape:
        raise ValueError("Model prediction and validation target shapes must match.")
    if not 0.0 < float(target_coverage) <= 1.0:
        raise ValueError("target_coverage must be in (0, 1].")
    if alpha_step <= 0.0 or alpha_max < 0.0:
        raise ValueError("alpha_step must be positive and alpha_max must be non-negative.")

    required_scores = np.zeros_like(mean)
    positive_residual = y_val > mean
    positive_std = std > 0.0
    scalable = positive_residual & positive_std
    required_scores[scalable] = (y_val[scalable] - mean[scalable]) / std[scalable]
    required_scores[positive_residual & ~positive_std] = np.inf

    target_count = int(np.ceil(float(target_coverage) * len(required_scores)))
    required_alpha = float(np.sort(required_scores)[target_count - 1])
    if np.isfinite(required_alpha):
        alpha = float(np.ceil(required_alpha / alpha_step) * alpha_step)
        coverage = float(np.mean(y_val <= mean + alpha * std))
        if alpha > alpha_max:
            warnings.warn(
                f"Required alpha {alpha:.3f} exceeds configured alpha_max "
                f"{alpha_max:.3f}; using the calibrated value to preserve "
                f"target coverage {target_coverage:.3f}.",
                RuntimeWarning,
            )
        return alpha, coverage

    alpha = float(alpha_max)
    coverage = float(np.mean(y_val <= mean + alpha * std))
    warnings.warn(
        f"Could not reach target coverage {target_coverage:.3f}: some validation "
        f"points have zero predictive std while their targets exceed the predicted "
        f"mean. Using alpha_max={alpha_max:.3f} with achieved coverage "
        f"{coverage:.3f}.",
        RuntimeWarning,
    )
    return alpha, coverage


class Survival_dual_ranking(Survival):
    def __init__(
        self,
        nds=None,
        crowding_func='cd',
        alpha_f1=1,
        alpha_f2=1,
        alpha=None,
        alphas=None,
    ):
        crowding_func_ = get_crowding_function(crowding_func)
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_
        self.alpha_f1 = alpha_f1
        self.alpha_f2 = alpha_f2
        self.alpha = alpha
        self.alphas = None if alphas is None else np.asarray(alphas, dtype=float)

    def _do(self, problem, pop, *args, random_state=None, n_survive=None, **kwargs):
        F = pop.get('F').astype(float, copy=False)
        if self.alpha is not None:
            if self.alpha == 0.8:
                F_upper = pop.get('F_q80').astype(float, copy=False)
            elif self.alpha == 0.9:
                F_upper = pop.get('F_q90').astype(float, copy=False)
            elif self.alpha == 0.95:
                F_upper = pop.get('F_q95').astype(float, copy=False)
            else:
                raise ValueError("alpha must be one of 0.8, 0.9, 0.95 for QR dual-ranking.")
            F_hybrid = np.concatenate([F, F_upper], axis=1)
        else:
            F_std = pop.get('std').astype(float, copy=False)
            alphas = (
                self.alphas
                if self.alphas is not None
                else np.array([self.alpha_f1, self.alpha_f2], dtype=float)
            )
            if alphas.shape != (F.shape[1],):
                raise ValueError(
                    f"Expected one dual-ranking alpha per objective; "
                    f"received {alphas.shape[0]} for {F.shape[1]} objectives."
                )
            F_upper = F + alphas * F_std
            F_hybrid = np.concatenate([F, F_upper], axis=1)
        fronts_hybrid = NonDominatedSorting().do(F_hybrid)

        survivors = []
        for k, front in enumerate(fronts_hybrid):
            I = np.arange(len(front))
            if len(survivors) + len(I) > n_survive:
                n_remove = len(survivors) + len(front) - n_survive
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=n_remove)
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy', random_state=random_state)
                I = I[:-n_remove]
            else:
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=0)

            for j, i in enumerate(front):
                pop[i].set('rank', k)
                pop[i].set('crowding', crowding_of_front[j])
            survivors.extend(front[I])
        return pop[survivors]
