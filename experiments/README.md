# Unified experiment suite

This directory is the single home for experiment configurations, notebooks,
executable per-method scripts, baselines, the complete sample-size runner, and
result summaries.

The vendored baseline tree tracks only the Python packages required by
Prob-RVEA, Prob-MOEA/D, and TGPR-MO. Upstream plots, generated results, sample
archives, documentation builds, and caches remain excluded from Git.

Run the complete configured experiment plan from the repository root:

```bash
python experiments/run_all.py --dry-run
python experiments/run_all.py --resume
```

The executable `Exp*.py` files were generated from their matching notebooks.

Run one experiment, for example:

```bash
python Exp1_GPR_RBF.py
```

Each script streams stdout/stderr to the terminal and appends the same content to `logs/<method_name>.log`.
After all configured problems finish, each script appends one row per method/optimizer/problem to:

- `results/results_real_world.csv`

Each script also appends one raw per-seed record per problem to:

- `results/<method_name>.txt`

`config.yaml` is the canonical default configuration. Standalone Exp1-Exp4
runs use `N=100` for every problem. Edit it to change problem lists, seeds,
population size, or sample sizes.

Standalone Exp11-Exp13 scripts have been removed. Their XGBoost, Weighted
Ensemble, and TabPFN surrogate implementations remain registered in
`sample_size_common.py` for the training-size sensitivity experiment.

The configured paper suite is ZDT1/2/3/4/6, OmniTest, VLMOP1-3, DTLZ1-7,
RE21-25, RE31-37, MO-Portfolio, and Molecule. Both two- and three-objective
problems are supported.


## Training-size sensitivity experiment

Defaults are stored in `config.yaml` under `sample_size_ablation`. The runner uses
the complete method registry in `sample_size_common.py`; CLI flags override
only the requested run.

The default design is:

- training sizes: `50, 100, 200, 400, 1000`;
- offline-data/LHS seeds: `1..10`;
- optimization seeds: `1..10`.
- TabPFN is disabled in the default plan; its implementation remains available
  for an explicit `--methods 'TabPFN + NSGA-II'` run.

```bash
python experiments/run_all.py --dry-run
python experiments/run_all.py --resume --max-workers 1
python experiments/run_all.py --resume --max-workers 72
python experiments/sample_size_summary.py
```

Methods run as complete stages in the configured/CLI method order.
`tabpfn_max_workers: 5` independently caps the TabPFN stage. With
`--max-workers 72`, every non-TabPFN method stage can use up to 72 workers;
when execution reaches TabPFN, that stage uses at most 5 workers.

Raw and summary CSV files are written under `results/csv/`.
Paired dataset archives are stored under `results/npz/`. Surrogate models are
not persisted as PKL files: each method trains one model per objective, uses them for all
optimizer seeds in that method/LHS/training-size group, and releases it after
the group results have been appended and flushed to CSV. AutoGluon uses
temporary model directories that are explicitly removed immediately after the
CSV write. At startup, obsolete
`surrogate_*.pkl` files and the legacy `AutogluonModels/` directory are removed.
After each method/LHS/training-size group, every worker runs Python garbage
collection and clears the PyTorch CUDA cache when CUDA is already in use.
After every complete, error-free main run, the runner automatically regenerates
all five summary CSV files from the accumulated raw result CSV files.
At startup, legacy `results/exp*_results.csv` rows missing from the new CSV
directory are copied into `results/csv/` without duplicating successful runs.
The resume plan then runs only experiment keys that do not have a successful
CSV row (or failed rows when `--retry-failed` is requested). Dataset NPZ files
are reused; models needed by incomplete groups are retrained.
A successful unique key is `(problem, method, training_size, lhs_seed, opt_seed)`.
Use `--retry-failed` to retry failed keys; successful keys are always skipped on resume.
Top-level Prob-RVEA and Prob-MOEA/D errors are isolated to their method group:
failed rows, including the traceback, are written to the corresponding problem
CSV and the worker continues with the next method.
TabPFN reads `TABPFN_PRIMARY_API_KEY` and optional fallback credentials from
environment variables. If the primary credential returns a quota/rate-limit/
HTTP-429 error during fitting or any optimization prediction, the worker
switches once and retries without logging either token. Never store live keys
in a YAML configuration file.

HV and IGD+ share the same min-max transform derived only from the offline
training objectives. HV uses the Xue et al. raw reference point after that
transform. IGD+ uses a true/reference front when supplied by the problem and
otherwise records an offline non-dominated-front fallback. The paper itself
reports HV only; IGD+ is an additional normalized metric in this repository.

All Dual Ranking uncertainty bounds use the configured one-sided quantile (0.90
by default). GPR exposes latent/epistemic standard deviation; q80, q90, and q95
use the matching Gaussian weights 0.8416, 1.2816, and 1.6449 instead of one
fixed standard-deviation multiplier. QR and BNN retain the original Dual
Ranking crossing rule: a crossed upper quantile is reflected as
`q50 + abs(q_upper-q50)`. No empirical coverage adjustment is applied.

Exp1–Exp4 train one surrogate per objective on 100% of the selected offline
dataset. The independent test data is reserved for prediction-error reporting.

## Official Off-MOO training-pool mode

`config_official_pool.yaml` enables the independent small-data mode. The
default `config.yaml` still uses the existing LHS path.

```bash
python experiments/run_all.py \
  --config experiments/config_official_pool.yaml \
  --dry-run

python experiments/run_all.py \
  --config experiments/config_official_pool.yaml \
  --resume
```

The official mode loads the untouched pools with
`off_moo_bench.make(task_name)` and copies `task.x`, `task.y`, `task.x_test`,
and `task.y_test`. It does not pass dataset-size or percentile filters to the
official API.

For each problem and offline seed, one permutation is cached under:

```text
data_subsets/{problem}/offline_seed_{seed}/
  permutation.npy
  indices_N50.npy
  indices_N100.npy
  indices_N200.npy
  indices_N400.npy
  indices_N1000.npy
  metadata.json
```

Every `indices_N*.npy` is a prefix of the same permutation, so the configured
offline datasets are nested. Here, `N` always means the number of selected
official training-pool rows; it is distinct from the optimizer population size,
which is configured separately as `pop_size=100`. Every surrogate and baseline
fits all N selected rows.
Optimization-seeded initial
populations are also drawn from the complete selected N rows. When N=50 and
`pop_size=100`, the deterministic draw uses replacement and preserves all 100
initial-population entries. The complete
official test pool is used only for prediction-reliability evaluation. For problems
without a true Pareto front, IGD+ uses the fixed non-dominated front of the
official training pool for evaluation only; it is not used by training,
surrogate normalization, uncertainty construction, or optimization. Offline/model
seed and optimization seed are recorded separately, and optimization-seeded
initial populations are shared across methods.

In official-pool mode, NSGA-II, Prob-RVEA, Prob-MOEA/D, TGPR-MO, and
DDMOEA-GAN all use the configured `n_gen` with the same convention: the initial
population is generation 1, followed by `n_gen - 1` offspring generations.
Prob-RVEA and TGPR-MO keep APD-selected solutions and deterministically fill
empty reference-vector slots by Pareto rank and crowding so the live population
remains exactly `pop_size`. Prob-MOEA/D uses exactly `pop_size` deterministic
reference directions and a rebuilt matching neighborhood matrix. With the
default `n_gen=100` and `pop_size=100`, every method is checked for exactly
10,000 surrogate evaluations. Raw result rows record both counters. Legacy LHS mode
retains the original 10,000-FE termination for the DESDEO baselines.
`protocol_version` is part of every resume/result key, so unversioned results
and results from the older uncertainty protocol are rerun instead of mixed.
Configured `n_gen` and `pop_size` are also part of that identity, so CLI budget
overrides cannot accidentally reuse results from another optimizer budget.

DDMOEA-GAN trains its stochastic WGAN and surrogate pool once per offline
dataset using `model_seed = offline_seed`. Its optimizer still uses the
critic-adjusted fitness internally, but final MSE/HV/IGD+ are evaluated from
the surrogate ensemble predictions at the final candidates, not from that
internal selection score. Two- and three-objective tasks use the corresponding
dynamic generator/discriminator dimensions.

HV and IGD+ normalization in official-pool mode uses fixed evaluation-only
bounds from the complete official training pool, making values comparable
across sample sizes, offline seeds, and methods. Those bounds are never used for
surrogate fitting or optimization normalization.
