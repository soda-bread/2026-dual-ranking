import contextlib
import io

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.core.callback import Callback

from src.real_world_problems import build_real_world_problem, is_real_world_problem
from src.problem_specs import canonical_problem_name, get_problem_spec


def _build_real_problem(problem_name, n_var=None, n_obj=None):
    pname = canonical_problem_name(problem_name)
    spec = get_problem_spec(pname)
    n_var = spec.n_var if n_var is None else int(n_var)
    n_obj = spec.n_obj if n_obj is None else int(n_obj)

    if is_real_world_problem(problem_name):
        return build_real_world_problem(problem_name)

    if "dtlz" in pname:
        return get_problem(pname, n_var=n_var, n_obj=n_obj)

    if "omnitest" in pname:
        return OmniTest(n_var=n_var)

    if pname.startswith("zdt"):
        return get_problem(pname, n_var=n_var)

    return get_problem(pname)


def build_problem(problem_name, n_var=None, n_obj=None):
    return _build_real_problem(problem_name, n_var=n_var, n_obj=n_obj)


def _quiet_predict(model, X):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return model.predict(X)

# Problem
class Benchmark_Problem(Problem):
    def __init__(
        self,
        model_f1=None,
        model_f2=None,
        n_var=None,
        n_obj=None,
        xl=None,
        xu=None,
        problem_name=None,
        use_surrogate=None,
        models=None,
    ):

        self.problem = _build_real_problem(problem_name, n_var=n_var, n_obj=n_obj)

        n_constr = self.problem.n_constr if self.problem.has_constraints() else 0

        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu,
                         n_constr=n_constr)

        if models is None:
            models = [model for model in (model_f1, model_f2) if model is not None]
        self.models = tuple(models)
        if use_surrogate is not None and len(self.models) != n_obj:
            raise ValueError(
                f"Expected {n_obj} surrogate models, received {len(self.models)}."
            )
        self.model_f1 = self.models[0] if self.models else None
        self.model_f2 = self.models[1] if len(self.models) > 1 else None
        self.use_surrogate = use_surrogate

    def _evaluate(self, X, out, *args, **kwargs):
        if self.use_surrogate == 'GPR_uncertainty':
          predictions = [model.predict(X) for model in self.models]
          out["F"] = np.column_stack([
              np.asarray(mean, dtype=float).reshape(-1)
              for mean, _ in predictions
          ])
          out["std"] = np.column_stack([
              np.asarray(std, dtype=float).reshape(-1)
              for _, std in predictions
          ])

          if self.problem.has_constraints():
            out["G"] = self.problem.evaluate(X, return_values_of=["G"])

        elif self.use_surrogate == 'BNN_uncertainty':
          predictions = [model.predict_distribution(X) for model in self.models]
          out["F"] = np.column_stack([pred[0] for pred in predictions])
          out["std"] = np.column_stack([pred[1] for pred in predictions])
          out["F_q80"] = np.column_stack([pred[2] for pred in predictions])
          out["F_q90"] = np.column_stack([pred[3] for pred in predictions])
          out["F_q95"] = np.column_stack([pred[4] for pred in predictions])

          if self.problem.has_constraints():
            out["G"] = self.problem.evaluate(X, return_values_of=["G"])

        elif self.use_surrogate == 'QR_uncertainty':
          df_test = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
          predictions = []
          for model in self.models:
              pred = model.predict(df_test)
              pred.columns = [f'y_q{q}' for q in pred.columns]
              predictions.append(pred)
          out["F"] = np.column_stack([pred['y_q0.5'].values for pred in predictions])
          out["F_q80"] = np.column_stack([pred['y_q0.8'].values for pred in predictions])
          out["F_q90"] = np.column_stack([pred['y_q0.9'].values for pred in predictions])
          out["F_q95"] = np.column_stack([pred['y_q0.95'].values for pred in predictions])

          if self.problem.has_constraints():
            out["G"] = self.problem.evaluate(X, return_values_of=["G"])

        elif self.use_surrogate == 'Autogluon':
          df_test = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
          out["F"] = np.column_stack([
              np.asarray(model.predict(df_test), dtype=float).reshape(-1)
              for model in self.models
          ])

          if self.problem.has_constraints():
            out["G"] = self.problem.evaluate(X, return_values_of=["G"])

        elif self.use_surrogate == 'TabPFN':
          X_test = np.asarray(X, dtype=float)
          out["F"] = np.column_stack([
              np.asarray(_quiet_predict(model, X_test), dtype=float).reshape(-1)
              for model in self.models
          ])

          if self.problem.has_constraints():
            out["G"] = self.problem.evaluate(X, return_values_of=["G"])

        else:
          out["F"] = self.problem.evaluate(X, return_values_of=["F"])


    
class EvaluatePreRealCallback(Callback):
    def __init__(self, true_problem, plot_every=1, use_opt=True, dynamic_show=False,
                 prefix="", obj_min=None, obj_max=None, hv_indicator=None):
        super().__init__()
        self.true_problem = true_problem
        self.plot_every = plot_every
        self.use_opt = use_opt
        self.dynamic_show = dynamic_show
        self.prefix = prefix
        self.max_f_so_far = None
        self.obj_min = None if obj_min is None else np.asarray(obj_min, dtype=float)
        self.obj_max = None if obj_max is None else np.asarray(obj_max, dtype=float)
        self.hv_indicator = hv_indicator
        self.records = []

        self.gen_list = []
        self.hv_sur_list = []
        self.hv_real_list = []

    def notify(self, algorithm):
        gen = algorithm.n_gen

        if gen != 1 and gen % self.plot_every != 0:
            return

        pop = algorithm.opt if self.use_opt else algorithm.pop
        X = pop.get("X")
        pre = pop.get("F")
        real = self.true_problem.evaluate(X, return_values_of=["F"])

        hv_sur = None
        hv_real = None
        if self.hv_indicator is not None and self.obj_min is not None and self.obj_max is not None:
            pre_norm = (pre - self.obj_min) / (self.obj_max - self.obj_min)
            real_norm = (real - self.obj_min) / (self.obj_max - self.obj_min)
            hv_sur = float(self.hv_indicator.do(pre_norm))
            hv_real = float(self.hv_indicator.do(real_norm))
        
        self.gen_list.append(gen)
        self.hv_sur_list.append(hv_sur)
        self.hv_real_list.append(hv_real)

        if self.dynamic_show:
            from IPython.display import clear_output

            clear_output(wait=True)

        max_pre = np.max(pre, axis=0)
        max_real = np.max(real, axis=0)
        max_f = np.maximum(max_pre, max_real)

        if self.max_f_so_far is None:
            self.max_f_so_far = max_f.copy()
        else:
            self.max_f_so_far = np.maximum(self.max_f_so_far, max_f)

        print(f"[{self.prefix}] Generation {gen}")
        for objective_index, maximum in enumerate(self.max_f_so_far, start=1):
            print(
                f"Max f{objective_index}: {maximum:.3f} | "
                f"offline max {self.obj_max[objective_index - 1]:.3f}"
            )

        if hv_sur is not None and hv_real is not None:
            print(f"HV sur : {hv_sur:.3f}")
            print(f"HV real: {hv_real:.3f}")

        result = evaluate_pre_real(
            pre,
            real,
            title=f"{self.prefix} | Gen {gen}",
            show_plot=True)

        if gen == 1 or gen == 100:
          _ = evaluate_pre_real(
            pre,
            real,
            show_plot=True,
            show_legend=False,       
            show_axis_labels=False,   
            point_size=50,
            tick_fontsize=22,
            save_svg=True,
            xlim=(-50, 600),
            ylim=(-50, 600),
            svg_path=f"figure_gen{gen}.svg",
        )

        self.records.append({
            "gen": gen,
            "X": X.copy(),
            "pre": pre.copy(),
            "real": real.copy(),
            "hv_sur": hv_sur,
            "hv_real": hv_real,
            **result
        })

def evaluate_pre_real(
    pre,
    real,
    title=None,
    figsize=(7, 6),
    point_size=20,
    tick_fontsize=12,
    label_fontsize=12,
    title_fontsize=14,
    legend_fontsize=11,
    show_plot=True,
    save_svg=False,
    svg_path="figure.svg",
    show_legend=True,       
    show_axis_labels=True,   
    x_label="F1",            
    y_label="F2",
    xlim=None,         
    ylim=None            
):
    pre = np.asarray(pre, dtype=float)
    real = np.asarray(real, dtype=float)

    if pre.ndim != 2 or real.ndim != 2:
        raise ValueError("pre and real must be 2D arrays.")
    if pre.shape[1] != real.shape[1]:
        raise ValueError("pre and real must have the same number of objectives.")
    if pre.shape[0] != real.shape[0]:
        raise ValueError("pre and real must have the same number of rows.")

    # row-wise Euclidean distance
    distances = np.sqrt(np.sum((pre - real) ** 2, axis=1))

    max_idx = np.argmax(distances)
    min_idx = np.argmin(distances)

    result = {
        "distances": distances,
        "max_distance": distances[max_idx],
        "max_obj_point": pre[max_idx],
        "max_f_real_point": real[max_idx],
        "min_distance": distances[min_idx],
        "min_obj_point": pre[min_idx],
        "min_f_real_point": real[min_idx],
        "mean_distance": np.mean(distances)
    }

    if (show_plot or save_svg) and pre.shape[1] == 2:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(pre.shape[0]):
            ax.annotate(
                '',
                xy=(real[i, 0], real[i, 1]),
                xytext=(pre[i, 0], pre[i, 1]),
                arrowprops=dict(
                    arrowstyle='->',
                    color='green',
                    lw=1.0,
                    alpha=0.8,
                    shrinkA=0,
                    shrinkB=0
                )
            )

        pre_label = 'pre' if show_legend else None
        real_label = 'real' if show_legend else None

        ax.scatter(
            pre[:, 0], pre[:, 1],
            color='#87CEEB',
            s=point_size,
            alpha=0.8,
            label=pre_label
        )

        ax.scatter(
            real[:, 0], real[:, 1],
            color='#FF7F0E',
            s=point_size,
            alpha=0.8,
            label=real_label
        )

        if show_axis_labels:
            ax.set_xlabel(x_label, fontsize=label_fontsize)
            ax.set_ylabel(y_label, fontsize=label_fontsize)

        if title is not None:
            ax.set_title(title, fontsize=title_fontsize)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.tick_params(axis='both', labelsize=tick_fontsize)

        if show_legend:
            ax.legend(fontsize=legend_fontsize)

        plt.tight_layout()

        if save_svg:
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Figure saved as SVG: {svg_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    elif show_plot or save_svg:
        print(
            f"Skipping 2D surrogate-vs-real plot for {pre.shape[1]} objectives."
        )

    print(
        f"Max:  {result['max_distance']:.3f}, "
        f"sur={np.array2string(result['max_obj_point'], precision=3)}, "
        f"real={np.array2string(result['max_f_real_point'], precision=3)}"
    )
    print(
        f"Min:  {result['min_distance']:.3f}, "
        f"sur={np.array2string(result['min_obj_point'], precision=3)}, "
        f"real={np.array2string(result['min_f_real_point'], precision=3)}"
    )
    print(f"Mean: {result['mean_distance']:.3f}")
    print("-" * 50)

    return result
