# Offline Data-Driven Multi-Objective Optimization

This repository contains experiments for offline data-driven multi-objective
optimization (MOO), using the task and metric settings in
`paper/xue24b.pdf`.

## Repository Layout

```text
experiments/           Original notebooks and shared recording utilities
external/offline-moo/  Vendored offline-moo benchmark source
results/               Benchmark result txt/csv files
src/                   Shared data generation, model, optimization, and metric code
```

## Benchmark Problem Protocol

The configured suite contains:

- two-objective ZDT1/2/3/4/6 and OmniTest;
- VLMOP1/2 and three-objective VLMOP3;
- three-objective DTLZ1-7;
- RE21-25, RE31-37, MO-Portfolio, and Molecule.

RE, VLMOP, MO-Portfolio, and Molecule use the implementations in
`external/offline-moo`; ZDT, DTLZ, and OmniTest use pymoo.

The true problem object is built through:

```python
from off_moo_bench.problem import get_problem
problem = get_problem("re21")
```

The corresponding source classes are in:

```text
external/offline-moo/off_moo_bench/problem/synthetic_func.py
```

These functions are explicit engineering-design benchmark formulas in the source code. In the offline optimization protocol, they are treated as black-box oracles: they are used to generate offline samples and to evaluate final solutions, but the optimizer itself uses only surrogate predictions.

### Offline Data Generation

The current experiment code does not load the official offline-moo `.npy` datasets for `re21`-`re25`. Instead, it generates offline data at runtime:

```python
X_train = sampling(problem, sample_size, seed=train_seed).get("X")
y_train = problem.evaluate(X_train, return_values_of=["F"])
```

This is implemented in:

```text
src/data.py
```

Current default settings are in:

```text
experiments/config.yaml
```

The training-size study uses sample sizes `50, 100, 200, 400, 1000`,
offline-data seeds `1..10`, and optimization seeds `1..10`.

The original offline-moo benchmark registers `RE21-Exact-v0` etc. with precomputed dataset shards such as:

```text
off_moo_bench/data/re21/re21-x-0.npy
off_moo_bench/data/re21/re21-y-0.npy
off_moo_bench/data/re21/re21-test-x-0.npy
off_moo_bench/data/re21/re21-test-y-0.npy
```

The default experiment scripts still use the existing LHS path. The independent
small-data configuration described below loads these official training and test
pools without quality/percentile filtering.

### True Evaluation

After surrogate optimization finishes, final candidate solutions are evaluated twice:

```python
obj = benchmark_problem_GPR.evaluate(solution, return_values_of=["F"])
f_real = problem.evaluate(solution, return_values_of=["F"])
```

- `obj`: surrogate-predicted objective values.
- `f_real`: true objective values from the offline-moo benchmark problem.

Metrics are then computed as:

- `MSE_sur_real`: MSE between `obj` and `f_real`.
- `HV_sur`: hypervolume computed using surrogate objectives.
- `HV_real`: hypervolume computed using true objectives.
- `IGD+_sur` and `IGD+_real`: IGD+ computed from surrogate and true objectives.

### Metric Protocol

For each offline dataset, objective values are min-max normalized with the
training objectives only:

```text
y_norm = (y - y_train_min) / (y_train_max - y_train_min)
```

The raw HV reference point reported by Xue et al. is transformed by the same
formula before constructing the HV indicator. No clipping is applied to final
solutions. The same normalization is used for IGD+. IGD+ uses the problem's
reference/true Pareto front when available. In LHS mode, a task without a true
front uses the current offline non-dominated front and records
`offline_non_dominated_front`. In official-pool mode, it instead uses one fixed
non-dominated front from the complete official training pool across every N,
seed, and method, recorded as
`official_training_pool_non_dominated_front`.

The reference paper reports HV, not IGD/IGD+, for its benchmark results because
true Pareto fronts are unavailable for many real tasks. IGD+ here is a
paper-compatible extension rather than a reproduced paper metric.

### Uncertainty Protocol

Dual Ranking uses the configured one-sided quantile (0.90 by default). GPR
predictions use latent function variance only (`include_likelihood=False`), so
fitted likelihood noise is not counted as epistemic uncertainty. Gaussian q80,
q90, and q95 candidates use `mean + z_q * epistemic_std`, with weights 0.8416,
1.2816, and 1.6449 respectively. BNN standard deviations and quantiles are
computed from posterior function-mean samples, excluding observation noise. QR
uses the model's native q80/q90/q95 directly; crossed quantiles are not reflected
or converted into heuristic upper bounds. No empirical coverage adjustment is
applied.

The standalone Exp1–Exp4 entry points train one surrogate per objective using
100% of the selected offline dataset. The independent test set is used only to
report prediction error.

An independent Off-MOO-Bench training-pool small-data mode is configured in
`experiments/config_official_pool.yaml`. It leaves the default LHS mode unchanged,
uses nested random prefixes for sample sizes 50/100/200/400/1000, and caches
the shared subset indices under `experiments/data_subsets/`. Here N is the selected
offline-dataset cardinality, while the optimizer population size is separately
fixed at 100. Every surrogate and baseline fits all N selected offline rows in
both standalone and unified runs.
Optimizer initial populations use the complete selected N rows. The default
official-pool optimization seeds are 1..10, matching the default LHS mode and
remaining independently controlled from offline/model seeds 1..10. When N=50
and `pop_size=100`, the seeded initializer samples selected rows with
replacement while preserving the configured 100-member initial population.

The official-pool configuration uses a common generation budget across
NSGA-II, Prob-RVEA, Prob-MOEA/D, TGPR-MO, and DDMOEA-GAN: generation 1 is the
initial population and the optimizer then executes `n_gen - 1` offspring
generations. Actual surrogate-evaluation counts are recorded separately because
Prob-RVEA and TGPR-MO deterministically fill empty APD reference-vector slots to
keep a live population of 100. Prob-MOEA/D likewise uses exactly 100 reference
directions and matching neighborhoods. Every official-pool run is checked for
100 generations, population size 100, and 10,000 surrogate evaluations.
Versioned result keys prevent resume from reusing results produced
under an older official-pool termination protocol; configured generation and
population budgets are included in the same identity.
The legacy LHS DESDEO termination remains unchanged at 10,000 counted surrogate
evaluations.

For `RE22`-`RE25`, the second objective in the offline-moo implementation is the sum of constraint violations, not a separate physical performance objective.

## Running Benchmark Experiments

The benchmark notebooks are in `experiments/`. Their standalone default uses
`N=100` for every configured problem. Edit `experiments/config.yaml` to change
problem names, seeds, population size, generations, or sample sizes.

Main notebooks include:

```text
Exp1_GPR_RBF.ipynb
Exp2_GPR_Matern.ipynb
Exp3_Autogluon_QR.ipynb
Exp4_BNN.ipynb
Exp5_Prob_RVEA_2022.ipynb
Exp6_Prob_MOEAD_2022.ipynb
Exp7_TGPR_MO_2023.ipynb
Exp8_DDMOEA_GAN_2024.ipynb
```

The XGBoost, Weighted Ensemble, and TabPFN surrogate implementations remain
available through `src/models.py` and the unified sample-size runner, but they
no longer have standalone Exp11-Exp13 notebooks. TabPFN is disabled in the
default run plan and can be enabled explicitly with `--methods`.

Run the complete default LHS plan from the repository root:

```bash
python experiments/run_all.py --dry-run
python experiments/run_all.py --resume
```

Use `experiments/config_official_pool.yaml` for the official-pool protocol.

Outputs are appended to:

```text
experiments/results/results_real_world.csv
experiments/results/<method_name>.txt
```

## Notes

- RE problems are real-world engineering design benchmarks with explicit
  analytical oracles.
- In benchmark experiments, the optimizer should not directly use the true oracle during optimization; the oracle is reserved for offline data generation and final evaluation.
- Molecule requires the optional scientific-design dependencies bundled by the
  upstream offline-moo project.
- Optional TabPFN credentials are read from environment variables; TabPFN is
  disabled by default.
