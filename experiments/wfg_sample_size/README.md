# OOF coverage-bias + Margin-U relation correction

This branch retains **OOF coverage-bias + Margin-U** as the best current
experimental candidate. “Best” means that it passed the frozen N=400 WFG
relation-F1 gate; it does not mean proven, optimizer-ready, or uniformly best.

The implementation is in `src/relation_correction.py`. All objectives are
minimized. Each fold's OOF predictions must come from models and preprocessing
that exclude the whole fold.

```python
from src.relation_correction import OOFCoverageMarginU

rule = OOFCoverageMarginU.from_oof(
    y_train, oof_mean, oof_latent_std, fold_ids
)
prediction = rule.predict_relations(
    full_model_mean, full_model_latent_std, pair_first, pair_second
)
relations = prediction.relation
```

The output uses `+1` when the first solution dominates, `-1` when the second
dominates, and `0` otherwise. It is a context-dependent pair classifier: its
call budget K depends on the supplied pair collection. It is not a globally
consistent `f_sur_adj`, and observed transitivity violations prevent treating
all its labels as ordinary Pareto comparisons of fixed objective vectors.

Run the self-contained checks with:

```bash
python -m unittest discover -s tests -p test_relation_correction.py
```

Read `VALIDITY_REPORT.md` and `EVIDENCE.json` before reuse. The experiments did
not run an optimizer and provide no HV or IGD+ evidence. No optimizer adapter
is intentionally included in this branch.
