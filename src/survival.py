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
                raw_upper = pop.get('F_q80').astype(float, copy=False)
            elif self.alpha == 0.9:
                raw_upper = pop.get('F_q90').astype(float, copy=False)
            elif self.alpha == 0.95:
                raw_upper = pop.get('F_q95').astype(float, copy=False)
            else:
                raise ValueError(
                    "alpha must be one of 0.8, 0.9, 0.95 for quantile dual-ranking."
                )
            if raw_upper.shape != F.shape:
                raise ValueError(
                    "Upper-quantile predictions must match the objective shape."
                )
            if not np.all(np.isfinite(raw_upper)):
                raise ValueError("Upper-quantile predictions must be finite.")
            # Use the model's quantile directly. Quantile crossing is not
            # reflected or otherwise converted into a heuristic upper bound.
            F_upper = raw_upper
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
