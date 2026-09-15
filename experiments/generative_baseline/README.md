# Generative Offline Multi-Objective Baselines

This directory integrates three methods unique to `2026-ICLR/model_generative`
into the unified dual-ranking experiment protocol: DOMOO, PCD, and ParetoFlow.
These are protocol adapters rather than complete copies of the original method
directories, datasets, evaluation scripts, and duplicate network definitions.

## Comparison Summary

| Aspect | ICLR `model_generative` | dual-ranking implementation |
| --- | --- | --- |
| Data | Each method loads, filters, or normalizes data independently | Fixed subsets and train/test splits from `src.official_pool` |
| Surrogate | DOMOO and ParetoFlow each include a separate multi-model surrogate | Reuses `MultipleModels-Vallina` from `experiments/DL_baseline` |
| ParetoFlow network | Method directory contains separate `FlowMatching` and `VectorFieldNet` implementations | Reuses the existing implementations in `external/offline-moo` |
| True function | Original scripts call it during their respective evaluation stages | Called once by the unified evaluator after final candidates are selected |
| Metrics | Commonly task min-max normalization, a `1.1` reference point, and D-best | Paper HV reference points, official/full-pool normalization, and IGD+ |
| Failures and reproducibility | Recorded independently by each method | Configuration and subset hashes, failure rows, independent model/optimization seeds, and resume support |

The adapters preserve the following core mechanisms:

- **DOMOO**: per-objective surrogates, energy-based confidence trained with
  Langevin negative samples, risk-suppressing Pareto-set learning, and joint
  selection from PSL and surrogate NSGA-II candidates.
- **PCD**: Pareto-rank and objective-density reweighting, objective-condition
  dropout, EDM-preconditioned loss, classifier-free guidance, and conditional
  sampling extrapolated toward the ideal point.
- **ParetoFlow**: unconditional flow matching, reference directions, and
  late-stage surrogate-gradient guidance.

PCD is a direct conditional generator without a separate objective surrogate.
Its `MSEpre`, `MSEsur_real`, `HVsur`, and `IGDplus_sur` values are therefore
NaN, while true HV and IGD+ are computed normally.

## Running the Baselines

Dependencies are shared with the existing DL baselines:

```bash
git submodule update --init external/offline-moo
python -m pip install -r experiments/generative_baseline/requirements.txt
```

Off-MOO task data must also exist in the expected `data/<task>/` directories.
Initializing only the code submodule without its data causes full runs to
report the missing `.npy` files explicitly.

Inspect the full experiment plan without loading PyTorch or data:

```bash
python experiments/generative_baseline/run.py --dry-run
```

Exercise the complete pipeline for all three methods with a minimal training
configuration:

```bash
python experiments/generative_baseline/run.py \
  --methods DOMOO,PCD,ParetoFlow \
  --problems zdt1 \
  --training-sizes 50 \
  --offline-seeds 1 \
  --optimization-seeds 1 \
  --device cpu \
  --smoke
```

Example full run for one task:

```bash
python experiments/generative_baseline/run.py \
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
