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

The implementation keeps the upstream model architecture and default
hyperparameters, while using this repository's common official-pool protocol:

- nested random-prefix subsets for `N=50/100/200/400/1000`;
- all selected `N` rows are used for model fitting (the official test pool is
  evaluation-only);
- offline/model seeds and optimization seeds are independent;
- neural methods use NSGA-II initialized from the selected offline subset;
- final candidates are evaluated by the true problem only after optimization;
- HV and IGD+ use the fixed full-training-pool normalization/reference data,
  matching `experiments/config_official_pool.yaml`.

The upstream defaults are 2048-2048 LeakyReLU MLPs, 200 epochs, batch size 128,
and a 100-generation/100-population NSGA-II run. MOBO keeps the upstream
qNEHVI acquisition defaults (128 MC samples, 256 raw samples, and 10 restarts),
but fits all N rows by default for consistency with the shared protocol. Set
`mobo.train_gp_data_size: 256` to reproduce the upstream GP-data cap. These are
expensive settings; use the smoke-test overrides first.

The shared protocol deliberately disables upstream COM data pruning so every
method receives the same selected `N` rows. COM's conservative loss, particle
updates, and dual-alpha update remain unchanged.

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
