# Offline Data-Driven Multi-Objective Optimization

This repository contains experiments for offline data-driven multi-objective optimization (MOO). It has two main parts:

- Benchmark experiments on engineering design problems such as `re21`-`re25`, `mo-portfolio`, `truss2d`, and `welded_beam`.
- A Case 2 building-space optimization problem using real building operation data.

## Repository Layout

```text
building_space_opt/    Case 2 building-space optimization notebooks and scripts
experiments/           Original notebooks and shared recording utilities
external/offline-moo/  Vendored offline-moo benchmark source
results/               Benchmark result txt/csv files
src/                   Shared data generation, model, optimization, and metric code
```

## Benchmark Problem Protocol

For `re21`-`re25`, the current code uses the benchmark problem implementations from `external/offline-moo`.

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

By default, `re21`-`re25` each use:

```yaml
sample_size: 300
train_seed: 42
test_seed: 1
val_size: 100
test_size: 100
```

The original offline-moo benchmark registers `RE21-Exact-v0` etc. with precomputed dataset shards such as:

```text
off_moo_bench/data/re21/re21-x-0.npy
off_moo_bench/data/re21/re21-y-0.npy
off_moo_bench/data/re21/re21-test-x-0.npy
off_moo_bench/data/re21/re21-test-y-0.npy
```

Those official data files are not used by the current scripts.

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

For `RE22`-`RE25`, the second objective in the offline-moo implementation is the sum of constraint violations, not a separate physical performance objective.

## Running Benchmark Experiments

The benchmark notebooks are in `experiments/`. Edit `experiments/config.yaml` to change problem names, seeds, population size, generations, or sample sizes.

Main notebooks include:

```text
Exp1_GPR_RBF_real_world_problem.ipynb
Exp2_GPR_Matern_real_world_problem.ipynb
Exp3_Autogluon_QR_real_world_problem.ipynb
Exp4_BNN_real_world_problem.ipynb
Exp11_Autogluon_XGBoost_real_world_problem.ipynb
Exp12_Autogluon_Ensemble_real_world_problem.ipynb
Exp13_TabPFN_3_real_world_problem.ipynb
```

Outputs are appended to:

```text
results/results_real_world.csv
results/<method_name>.txt
```

## Case 2 Building-Space Optimization

The Case 2 problem is in:

```text
building_space_opt/
```

Main scripts:

```text
Case_2_TabPFN_NSGA_II.py
Case_2_GPR_Matern_NSGA_II.py
Case_2_GPR_Matern_Dual_Ranking_NSGA_II.py
Case_2_initial_objectives.py
```

The scripts resolve the dataset from one of these locations:

```text
/content/drive/MyDrive/2026 Real-wrold problem/building_space_opt/Dataset
/rds/projects/w/wangsu-building-automation/Huanbo/2026_real_world_problem/building_space_opt/Dataset
building_space_opt/Dataset
```

The training data uses the first day of `data_office_1.csv`, i.e. the first 288 rows.

Run on a server:

```bash
cd /rds/projects/w/wangsu-building-automation/Huanbo/2026_real_world_problem/building_space_opt
python -u Case_2_GPR_Matern_NSGA_II.py
```

Run in Colab by opening the corresponding notebook:

```text
Case_2_TabPFN_NSGA_II.ipynb
Case_2_GPR_Matern_NSGA_II.ipynb
Case_2_GPR_Matern_Dual_Ranking_NSGA_II.ipynb
```

### Case 2 Outputs

Each Case 2 method writes outputs to:

```text
building_space_opt/outputs/
```

Expected files per method:

```text
<method>_optimal_solutions.txt
<method>_pareto_front.png
<method>_pareto_front_zoom.png
<method>_layout_ABC.png
```

`Case_2_initial_objectives.py` writes:

```text
outputs/case_2_initial_objectives.txt
```

## Notes

- `re21`-`re25` are best described as real-world engineering design benchmark problems with explicit analytical oracles.
- The Case 2 building-space problem is a real data-driven problem using measured building operation data.
- In benchmark experiments, the optimizer should not directly use the true oracle during optimization; the oracle is reserved for offline data generation and final evaluation.
- TabPFN scripts require a valid `TABPFN_TOKEN` environment variable or an already configured TabPFN client.
