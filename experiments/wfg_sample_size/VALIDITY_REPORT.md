# Validity assessment

## Decision

Keep OOF coverage-bias + Margin-U as the current **experimental relation
candidate**, but do not wire it into an optimizer yet. The evidence supports a
relation-F1 improvement on the tested WFG random pools. It does not establish a
safe Pareto-order correction, a globally adjusted objective vector, or an
optimization benefit.

## Method

Five-fold out-of-fold predictions estimate two training-only quantities:

1. objective-wise uncertainty scale,
   `c_j = RMS(OOF residual_j) / RMS(OOF latent std_j)`;
2. dominance-coverage bias,
   `b = r_true_OOF - r_predicted_OOF`.

For P candidate pairs, the rule requests

`K = round(P * clip(r_centre_pool + b, 0, 1))`

dominance calls. For a possible A-dominates-B direction it computes

`z_j = (mu_Bj - mu_Aj) / (c_j * sqrt(std_Aj^2 + std_Bj^2))`

and ranks the two directions by the sum of `log Phi(z_j)`. It retains the K
highest-scoring non-tied directions. Candidate truth is never an input.

This uses an independent-endpoint approximation; posterior covariance between
A and B is not available in the repository prediction interface. The score is
therefore not described as a calibrated dominance probability.

## Positive evidence

At N=400, Macro-F1 and directional dominance-F1 improved over the centre,
same-K margin-only and five candidate-wise uncertainty permutations for both
RBF and Matérn on all four WFG problems after averaging three seeds. The exact
increments are in `EVIDENCE.json`.

A missing component ablation was added post hoc: keep Margin-U scores fixed but
use the centre method's number of dominance calls. Adding the OOF coverage
correction improved Macro-F1 by 0.006518 for Matérn and 0.018371 for RBF, with
3/4 problem wins for each. Its estimated correction direction matched the
evaluation-pool direction in all 24 N=400 problem-kernel-seed cases, and reduced
absolute coverage error in 21/24.

## Why the evidence is not conclusive

Candidate-specific uncertainty contributed little relative to a stronger
constant-uncertainty control. Replacing every candidate's standard deviation by
the objective-wise pool RMS reduced Macro-F1 by only 0.000601 for Matérn and
0.000531 for RBF; the original won only 14/24 seed cases. Consequently, beating
a permutation control does not show that most of the gain came from correctly
attached candidate-wise uncertainty.

The apparent gain is also a precision-recall tradeoff. Against centre
predictions at N=400, Macro-F1 improved in 24/24 cases but exact accuracy in
17/24. Dominance precision fell in 18/24, while recall increased. Across both
kernels the rule added about 1,719 false dominance calls per 200,000 evaluated
pairs on average. This matters if a false dominance decision can eliminate a
good candidate.

The output is not a standard Pareto order. Among triangles for which all three
pairs happened to be evaluated, N=400 produced 106 transitivity violations in
6,227 directed two-step paths, across 12/24 cases: A dominated B and B dominated
C, while A did not dominate C. A fixed vector-valued `f_sur_adj` under ordinary
strict Pareto dominance cannot reproduce all these labels. No directed
three-cycle was observed in the sampled triangles, but incomplete pair sampling
cannot establish acyclicity.

OOF-to-pool transfer remains an assumption. OOF models use less training data
than the final model, and both training and evaluation pools were uniform. An
optimizer-induced candidate distribution may differ. The four benchmarks also
belong to one WFG family; the 200,000 dependent pairs per problem are not
independent experimental replicates.

## Required next gate

Freeze this implementation and validate on an unseen problem family with the
same centre, centre-K U, raw OOF-prior K, same-K margin-only,
constant-uncertainty and permutation controls. Report exact accuracy,
dominance precision/recall, false dominance count and transitivity alongside
F1. Only after passing that gate should a relation-graph survivor that explicitly
handles inconsistency be designed and tested with HV/IGD+.

This assessment is post-hoc and does not redefine the original exploratory
gate. No optimizer was run, and no hyperparameter was selected from these
outcomes.
