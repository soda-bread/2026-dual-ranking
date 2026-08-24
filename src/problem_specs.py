"""Problem dimensions and HV reference points from Xue et al. (2024).

The source values are reported in Tables 3 and 6 and Appendix B of
``paper/xue24b.pdf``.  Molecule is treated as two-objective because Appendix
B.5, its reported reference point, and the bundled Off-MOO implementation are
all two-objective (despite Table 1 listing three objectives).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    n_var: int
    n_obj: int
    reference_point: tuple[float, ...]
    family: str


PROBLEM_SPECS = {
    "zdt1": ProblemSpec(30, 2, (1.10, 8.58), "synthetic"),
    "zdt2": ProblemSpec(30, 2, (1.10, 9.59), "synthetic"),
    "zdt3": ProblemSpec(30, 2, (1.10, 8.74), "synthetic"),
    "zdt4": ProblemSpec(10, 2, (1.10, 300.42), "synthetic"),
    "zdt6": ProblemSpec(10, 2, (1.07, 10.27), "synthetic"),
    "omnitest": ProblemSpec(2, 2, (2.40, 2.40), "synthetic"),
    "vlmop1": ProblemSpec(1, 2, (4.0, 4.0), "synthetic"),
    "vlmop2": ProblemSpec(6, 2, (1.10, 1.10), "synthetic"),
    "vlmop3": ProblemSpec(2, 3, (9.07, 66.62, 0.23), "synthetic"),
    "dtlz1": ProblemSpec(7, 3, (558.21, 552.30, 568.36), "synthetic"),
    "dtlz2": ProblemSpec(10, 3, (2.77, 2.78, 2.93), "synthetic"),
    "dtlz3": ProblemSpec(10, 3, (1703.72, 1605.54, 1670.48), "synthetic"),
    "dtlz4": ProblemSpec(10, 3, (3.03, 2.83, 2.78), "synthetic"),
    "dtlz5": ProblemSpec(10, 3, (2.65, 2.61, 2.70), "synthetic"),
    "dtlz6": ProblemSpec(10, 3, (9.80, 9.78, 9.78), "synthetic"),
    "dtlz7": ProblemSpec(10, 3, (1.10, 1.10, 33.43), "synthetic"),
    "re21": ProblemSpec(4, 2, (3144.44, 0.05), "real"),
    "re22": ProblemSpec(3, 2, (829.08, 2407217.25), "real"),
    "re23": ProblemSpec(4, 2, (713710.88, 1288669.78), "real"),
    "re24": ProblemSpec(2, 2, (5997.83, 43.67), "real"),
    "re25": ProblemSpec(3, 2, (124.79, 10038735.00), "real"),
    "re31": ProblemSpec(3, 3, (808.85, 6893375.82, 6793450.00), "real"),
    "re32": ProblemSpec(4, 3, (290.66, 16552.46, 388265024.00), "real"),
    "re33": ProblemSpec(4, 3, (8.01, 8.84, 2343.30), "real"),
    "re34": ProblemSpec(5, 3, (1702.52, 11.68, 0.26), "real"),
    "re35": ProblemSpec(7, 3, (7050.79, 1696.67, 397.83), "real"),
    "re36": ProblemSpec(4, 3, (10.21, 60.00, 0.97), "real"),
    "re37": ProblemSpec(4, 3, (0.99, 0.96, 0.99), "real"),
    "mo-portfolio": ProblemSpec(20, 2, (0.29, -0.13), "real"),
    "molecule": ProblemSpec(32, 2, (0.09, 0.04), "real"),
}


PROBLEM_ALIASES = {
    "mo_portfolio": "mo-portfolio",
    "portfolio": "mo-portfolio",
    "portfolio-exact-v0": "mo-portfolio",
    "molecule-exact-v0": "molecule",
    **{
        f"re{number}-exact-v0": f"re{number}"
        for number in (*range(21, 26), *range(31, 38))
    },
}


BENCHMARK_PROBLEMS = tuple(
    name for name, spec in PROBLEM_SPECS.items() if spec.family == "synthetic"
)
REAL_WORLD_PROBLEMS = tuple(
    name for name, spec in PROBLEM_SPECS.items() if spec.family == "real"
)
EXPERIMENT_PROBLEMS = BENCHMARK_PROBLEMS + REAL_WORLD_PROBLEMS


def canonical_problem_name(problem_name):
    name = str(problem_name).strip().lower().replace("_", "-")
    return PROBLEM_ALIASES.get(name, name)


def get_problem_spec(problem_name):
    name = canonical_problem_name(problem_name)
    try:
        return PROBLEM_SPECS[name]
    except KeyError as error:
        raise ValueError(f"Problem settings are not configured for '{problem_name}'.") from error


def get_paper_reference_point(problem_name):
    return np.asarray(get_problem_spec(problem_name).reference_point, dtype=float)
