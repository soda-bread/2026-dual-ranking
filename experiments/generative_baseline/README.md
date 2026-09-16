# Generative Offline Multi-Objective Baselines

These methods belong only to the ten-baseline entry
`.venv/bin/python experiments/run_baselines.py`. This directory's `run.py`
remains the direct entry point for generative-only runs.

This directory contains repository-local adapters for PCD and ParetoFlow.
Their implementations and runtime do not import from the earlier comparison
project or any other sibling checkout.
These are protocol adapters rather than complete copies of the original method
directories, datasets, evaluation scripts, and duplicate network definitions.

## Comparison Summary

| Aspect | ICLR `model_generative` | dual-ranking implementation |
| --- | --- | --- |
| Data | Each method loads, filters, or normalizes data independently | Fixed subsets and train/test splits from `src.official_pool` |
| Surrogate | ParetoFlow uses independent objective MLPs with exponential LR decay and validation-PCC checkpoints | Reuses the shared MLP/scaler/predictor components but follows ParetoFlow's own 0.98 LR-decay and validation-PCC checkpoint protocol by default; `trainer: dl_baseline` restores the previous trainer |
| ParetoFlow network | Method directory contains separate `FlowMatching` and `VectorFieldNet` implementations | Reuses the existing implementations in `external/offline-moo` |
| True function | Original scripts call it during their respective evaluation stages | Called once by the unified evaluator after final candidates are selected |
| Metrics | Commonly task min-max normalization, a `1.1` reference point, and D-best | Paper HV reference points, official/full-pool normalization, and IGD+ |
| Failures and reproducibility | Recorded independently by each method | Configuration and subset hashes, failure rows, independent model/optimization seeds, and resume support |

The adapters preserve the following core mechanisms:

- **PCD**: dominance-count and objective-bin reweighting, objective-condition
  dropout, EDM-preconditioned loss, classifier-free guidance, and
  reference-direction extrapolation beyond the observed Pareto front.
- **ParetoFlow**: unconditional flow matching and the complete upstream
  `FlowMatching.paretoflow_sample` procedure, including offspring generation,
  neighborhood and angle filtering, archive updates, D-best initialization,
  and late-stage surrogate-gradient guidance.

PCD is a direct conditional generator without a separate objective surrogate.
Its `MSEpre`, `MSEsur_real`, `HVsur`, and `IGDplus_sur` values are therefore
NaN, while true HV and IGD+ are computed normally.

## Fidelity and Protocol Adaptations

ParetoFlow directly calls the implementation vendored under
`external/offline-moo`. Its task adapter exposes only the selected
official-pool subset: D-best initialization comes from that subset, and
intermediate predictions and internal diagnostic HV use the shared surrogate
instead of the true oracle. Final true objectives are still queried only by
the unified evaluator. The returned archive is repaired against the true task
bounds, scored by the shared surrogate, and reduced to the requested output
size by rank and crowding. Flow training draws mini-batches with replacement
for at most 10,000 optimizer steps. Early stopping is retained, but cannot stop
training before 1,000 steps. To make validation comparable on validation sets
as small as 10 rows, its CFM loss reuses fixed prior samples, time samples, and
path noise (common random numbers) at every checkpoint; it is evaluated every
100 steps, averaged over four fixed draws, and the best checkpoint is restored.
Upstream instead validates with a much more expensive NLL based on `log_prob`,
1000-step reverse Euler, and a Hutchinson trace estimate over a much larger
dataset. If there are fewer than twice `min_validation_rows`, validation and
early stopping are disabled and all configured steps run. The proxy still
reserves a proportional split, decays its learning rate by 0.98 each epoch, and
reloads the checkpoint with the highest validation PCC. Setting
`proxy.trainer: dl_baseline` restores the previous cosine-scheduled
no-validation proxy behavior.

PCD follows the official dominance-count weighting, fixed 30-bin density
weighting, residual MLP denoiser, EMA sampling weights, cosine learning-rate
schedule, AdamW parameter groups, and reference-direction conditioning in
z-score objective space. It uses up to 32 energy reference directions with the
upstream fixed seed 42; for 100 outputs, the associated base points are tiled
with ceiling division and truncated rather than reducing the direction count
to 25. EMA uses the official 100-step copy warmup, 10-step update interval,
and inverse-power decay capped at 0.995. Its D-best source is
the leading 256 solutions (or the complete set when smaller) selected from the
official-pool subset by non-dominated rank and crowding. Synthetic tasks use
the published synthetic configuration; RE21--RE37 use the wider four-block
network and RE-specific training and churn values from `config/re.gin`.
`molecule` and `mo-portfolio` intentionally retain the synthetic configuration;
the former does not use upstream `scientific.gin` in this comparison, and the
latter is outside the original PCD task suite. The
official 80/20 split is not used because validation is logging-only and would
unnecessarily reduce these small training subsets.

Two minor training differences are retained for the small-data protocol:
mini-batches are drawn with replacement on every step instead of iterating
through epoch-wise shuffled data, and LayerNorm weights are excluded from
weight decay by the local parameter-name grouping (the upstream grouping may
decay `ln.weight`).

The common protocol intentionally requests 100 outputs rather than the
upstream 256, fits x/y standardization on the selected official-pool subset
rather than the full dataset, and applies final task-bound clipping plus the
repository's portfolio repair. PCD conditions are not clipped in objective
space. Protocol v2 invalidates results produced by the earlier simplified
ParetoFlow and PCD adapters, so those rows must be rerun.

ParetoFlow rows created under the earlier epoch/CFM-early-stopping protocol are
invalidated automatically by the configuration hash. The fixed PCD reference
direction seed likewise changes PCD's configuration hash, so older PCD rows are
not resumed. The shared protocol version is intentionally not increased
because invalidation is method-specific.

## Running the Baselines

Dependencies are shared with the existing DL/MOBO baselines:

```bash
git submodule update --init external/offline-moo
python3.11 scripts/setup_environment.py
```

Before a formal run, download each task's official `.npy` pool into
`external/offline-moo/data/<task>/`. Every pool used with `N=1000` must contain
at least 1,000 training rows; the official released pools satisfy this.

Independent `(method, problem, N, offline_seed)` groups can use multiple CPU
workers while optimization seeds within a group reuse one fitted model:

```bash
.venv/bin/python experiments/generative_baseline/run.py \
  --resume \
  --device cpu \
  --max-workers 72
```

The parent process serializes CSV writes, and the effective worker count is
capped by the number of pending groups. Multi-worker execution is intended for
CPU runs; using many processes against one GPU can exhaust device memory.

Off-MOO task data must also exist in the expected `data/<task>/` directories.
Initializing only the code submodule without its data causes full runs to
report the missing `.npy` files explicitly.

Inspect the full experiment plan without loading PyTorch or data:

```bash
python3 experiments/generative_baseline/run.py --dry-run
```

Exercise the complete pipeline for both methods with a minimal training
configuration:

```bash
.venv/bin/python experiments/generative_baseline/run.py \
  --methods PCD,ParetoFlow \
  --problems zdt1 \
  --training-sizes 50 \
  --offline-seeds 1 \
  --optimization-seeds 1 \
  --device cpu \
  --smoke
```

Example full run for one task:

```bash
.venv/bin/python experiments/generative_baseline/run.py \
  --methods PCD \
  --problems zdt1 \
  --training-sizes 1000 \
  --offline-seeds 1 \
  --optimization-seeds 1
```

Defaults are defined in `config.yaml`. Summary rows are written to
`results/generative_baselines.csv`, and final candidates are stored in
`results/candidates/<method>/*.npz`. Successful rows are resumed by default;
use `--no-resume` to force a rerun.

## Implementation Scope

All 30 current benchmark tasks use fixed-length continuous decision
representations supported by this directory. Portfolio candidates continue to
pass through the repository's existing repair logic. The adapters do not copy
wandb or gin integration, task duplicates, plotting scripts, result
post-processing, surrogate duplicates, or ParetoFlow network duplicates from
the ICLR directory. This avoids having two definitions of the same protocol
drift apart over time.
