# Off-MOO deep-learning baselines

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
- all selected `N` rows are used for model fitting instead of reserving 10% for
  checkpoint selection (the official test pool is evaluation-only);
- offline/model seeds and optimization seeds are independent;
- neural methods use upstream NDS-ordered NSGA-II initialization; when
  `N < pop_size`, LHS samples from the true problem bounds fill the shortage.
  This is a deliberate deviation: upstream's `[0, 1]` space is the dataset
  min-max envelope and includes its test pool;
- final candidates are evaluated by the true problem only after optimization;
- HV and IGD+ use the fixed full-training-pool normalization/reference data,
  matching `experiments/config_official_pool.yaml`.

The upstream defaults are 2048-2048 LeakyReLU MLPs, 200 epochs, batch size 128,
and a 50-generation/256-population NSGA-II run. MOBO keeps the upstream
non-dominated-first 256-row GP cap, one qNEHVI proposal batch, 128 MC samples,
256 raw samples, and 10 restarts. For small datasets, the GP cap becomes
`min(256, N)`. When the last admitted Pareto front crosses that cap, upstream
truncates it by crowding distance, while this adapter keeps its front order.

The qNEHVI reference point starts from the same raw-scale `1.1 * nadir` as
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
python -m pip install -r experiments/DL_baseline/requirements.txt
```

The pinned packages match the legacy upstream environment and are best used
with Python 3.8. Scientific-design tasks such as `molecule` also require the
task-specific dependencies and data described by the upstream project.

## Usage

Inspect the complete plan without importing PyTorch, Pymoo, or BoTorch:

```bash
python experiments/DL_baseline/run.py --dry-run
```

Small End2End smoke run:

```bash
python experiments/DL_baseline/run.py \
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
python experiments/DL_baseline/run.py \
  --methods End2End-Vallina,MultipleModels-Vallina,MultipleModels-COM,MOBO-Vallina \
  --problems zdt1 \
  --training-sizes 100 \
  --offline-seeds 1 \
  --optimization-seeds 1
```

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
