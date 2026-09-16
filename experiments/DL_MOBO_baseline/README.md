# Off-MOO DL and MOBO baselines

These methods belong only to the ten-baseline entry
`.venv/bin/python experiments/run_baselines.py`. This directory's `run.py`
remains the direct entry point for DL/MOBO-only runs.

This directory adapts four methods from
[`lamda-bbo/offline-moo`](https://github.com/lamda-bbo/offline-moo):

- `End2End-Vallina`: one MLP jointly predicts every objective;
- `MultipleModels-Vallina`: one independently trained MLP per objective;
- `MultipleModels-COM`: one conservative objective model per objective using
  particle negatives and the COM Lagrangian;
- `MOBO-Vallina`: a multi-output exact GP with one qNEHVI proposal batch.

`Vallina` intentionally retains the spelling used by the upstream repository.
The implementation was checked against upstream commit
`1f62a521a171b94ab90e73715a272931f42a1395`, which is the submodule revision
recorded by this repository.

## Protocol

The implementation follows the upstream model, loss, optimizer, initialization,
and default search budget, while adapting data handling to this repository's
small official-pool subsets:

- nested random-prefix subsets for `N=50/100/200/400/1000`;
- all selected `N` rows are used for model fitting. Upstream creates a 90/10
  train/validation split and reports validation metrics, but the same run does
  not reload a validation-selected checkpoint; the official test pool remains
  evaluation-only here;
- offline/model seeds and optimization seeds are independent;
- neural methods use upstream NDS-ordered NSGA-II initialization; when
  `N < pop_size`, LHS samples fill the shortage;
- NSGA-II searches the true task bounds `[xl, xu]`. Upstream searches `[0, 1]`
  after mapping through a dataset min-max envelope that includes its test pool.
  The true-bound search keeps a common feasible domain across methods and
  avoids test-set leakage;
- final candidates are evaluated by the true problem only after optimization;
- HV and IGD+ use the fixed full-training-pool normalization/reference data,
  matching `experiments/config_official_pool.yaml`.

The upstream neural defaults remain 2048-2048 LeakyReLU MLPs, 200 epochs, and
batch size 128. The comparison budget is intentionally changed from upstream's
50-generation/256-population run to 100 generations with population 100, so
every method submits 100 solutions and neural NSGA-II methods use 10,000
surrogate evaluations. MOBO therefore requests q=100 instead of q=256. It
keeps the upstream non-dominated-first 256-row GP training cap, 128 MC samples,
256 raw samples, and 10 restarts; for small datasets, the GP cap becomes
`min(256, N)`. All DL/MOBO methods, including MOBO, use the same configured
optimization seeds. This repository's non-dominated truncation differs from upstream
`get_N_nondominated_index`: when the last Pareto front crosses the cap, the
upstream helper mistakenly applies a front-local index as a global row index,
whereas this adapter keeps the correctly addressed front order.

MOBO standardizes each GP objective with a z-score, whereas upstream applies a
min-max transform. The qNEHVI reference point starts from the same raw-scale `1.1 * nadir` as
upstream, then passes through the same objective z-score transform as the GP
targets. This fixes an upstream scale mismatch in which a raw reference point
is compared with normalized objectives. Likewise, this implementation scales
designs once using the true problem bounds. Upstream first min-max normalizes
them with the full dataset envelope (including its test pool) and then applies
problem-bound normalization inside MOBO; on RE tasks whose bounds are not
`[0, 1]`, that double normalization can map candidates outside the bounds.
`ensure_dominated_reference: true` moves only invalid reference coordinates
below the observed maximization values so qNEHVI remains defined on small
subsets.

The small-data adaptation deliberately disables upstream COM's additional 20%
Pareto pruning so every method receives all selected `N` rows. COM keeps the
same conservative loss, particle updates, and dual-alpha update; only the
pruning-associated `1 / 0.2` global loss multiplier is omitted with pruning.

## Installation

From the repository root:

```bash
git submodule update --init external/offline-moo
python3.11 scripts/setup_environment.py
```

The DL requirements redirect to the repository's single Python 3.11 dependency
set. Historical upstream pins are retained only as provenance in the vendored
source and must not be installed over this environment. Scientific-design
tasks such as `molecule` additionally need their repository-owned task data.

## Usage

Inspect the complete plan without importing PyTorch, Pymoo, or BoTorch:

```bash
python3 experiments/DL_MOBO_baseline/run.py --dry-run
```

Small End2End smoke run:

```bash
.venv/bin/python experiments/DL_MOBO_baseline/run.py \
  --methods End2End-Vallina \
  --problems zdt1 \
  --training-sizes 50 \
  --offline-seeds 1 \
  --optimization-seeds 1 \
  --epochs 2 \
  --n-gen 2 \
  --pop-size 10 \
  --device cpu
```

Run all four methods for one paired subset:

```bash
.venv/bin/python experiments/DL_MOBO_baseline/run.py \
  --methods End2End-Vallina,MultipleModels-Vallina,MultipleModels-COM,MOBO-Vallina \
  --problems zdt1 \
  --training-sizes 100 \
  --offline-seeds 1 \
  --optimization-seeds 1
```

Run independent model/data groups on up to 72 CPU workers:

```bash
.venv/bin/python experiments/DL_MOBO_baseline/run.py \
  --resume \
  --device cpu \
  --max-workers 72
```

One worker owns a complete `(method, problem, N, offline_seed)` group and
reuses its fitted model for all requested optimization seeds. Result rows are
returned to and written by the parent process. The effective worker count is
capped by the number of groups.

Successful runs are skipped by default. Use `--no-resume` to run them again.
The summary is appended to `results/dl_baselines.csv`; raw candidates and their
surrogate/true objectives are stored under `results/candidates/`.

Protocol v3 fixes the LHS-fill seed used by neural methods when `N < 256`.
Existing neural results for `N=50/100/200` must therefore be rerun. Neural
results for `N=400/1000` and all MOBO results are numerically unaffected, but
the v3 resume key intentionally treats their v2 rows as incomplete too. To
retain those unaffected rows, change only their `protocol_version` field in the
CSV from `off_moo_dl_baselines_official_pool_v2` to
`off_moo_dl_baselines_official_pool_v3` before resuming.
