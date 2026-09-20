"""Repository-local interpreter and dependency checks for experiment runners."""

from __future__ import annotations

import importlib
import importlib.util
from importlib import metadata
import os
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_VENV = REPO_ROOT / ".venv"
PROJECT_PYTHON = PROJECT_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
OFFLINE_MOO_ROOT = REPO_ROOT / "external" / "offline-moo"

BASE_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
    "pymoo": "pymoo",
    "optproblems": "optproblems",
    "diversipy": "diversipy",
}

MOLECULE_MODULES = {
    "RDKit (required by Molecule)": "rdkit",
    "Chemprop (required by Molecule)": "chemprop",
}

PINNED_VERSIONS = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.12.0",
    "scikit-learn": "1.5.2",
    "PyYAML": "6.0.2",
    "pymoo": "0.6.1.6",
    "optproblems": "1.3",
    "graphviz": "0.20.3",
    "diversipy": "0.8",
    "GPy": "1.13.2",
    "statsmodels": "0.14.4",
    "pyDOE2": "1.3.0",
    "torch": "2.2.2",
    "botorch": "0.10.0",
    "gpytorch": "1.11",
    "pyro-ppl": "1.9.1",
    "rdkit": "2023.9.4",
    "chemprop": "1.6.1",
    "autogluon.tabular": "1.2.0",
    "xgboost": "2.1.3",
}

MODULE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "pymoo": "pymoo",
    "optproblems": "optproblems",
    "graphviz": "graphviz",
    "diversipy": "diversipy",
    "GPy": "GPy",
    "statsmodels": "statsmodels",
    "pyDOE2": "pyDOE2",
    "torch": "torch",
    "botorch": "botorch",
    "gpytorch": "gpytorch",
    "pyro": "pyro-ppl",
    "rdkit": "rdkit",
    "chemprop": "chemprop",
    "autogluon.tabular": "autogluon.tabular",
    "xgboost": "xgboost",
}

METHOD_MODULES = {
    "GPR-RBF + NSGA-II": {"GPy": "GPy"},
    "GPR-RBF + NSGA-II + DR": {"GPy": "GPy"},
    "GPR-Matern + NSGA-II": {"GPy": "GPy"},
    "GPR-Matern + NSGA-II + DR": {"GPy": "GPy"},
    "QR + NSGA-II": {"AutoGluon": "autogluon.tabular"},
    "QR + NSGA-II + DR": {"AutoGluon": "autogluon.tabular"},
    "BNN + NSGA-II": {"PyTorch": "torch", "Pyro": "pyro"},
    "BNN + NSGA-II + DR": {"PyTorch": "torch", "Pyro": "pyro"},
    "XGBoost + NSGA-II": {
        "AutoGluon": "autogluon.tabular",
        "XGBoost": "xgboost",
    },
    "WeightedEnsemble L2 + NSGA-II": {"AutoGluon": "autogluon.tabular"},
    "TGPR-MO": {
        "GPy": "GPy",
        "Graphviz": "graphviz",
        "optproblems": "optproblems",
        "pyDOE2": "pyDOE2",
    },
    "DDMOEA-GAN": {"PyTorch": "torch"},
    "Prob-RVEA": {
        "optproblems": "optproblems",
        "statsmodels": "statsmodels",
        "pyDOE2": "pyDOE2",
    },
    "Prob-MOEA/D": {
        "optproblems": "optproblems",
        "statsmodels": "statsmodels",
        "pyDOE2": "pyDOE2",
    },
    "TabPFN + NSGA-II": {"tabpfn-client": "tabpfn_client"},
    "End2End-Vallina": {"PyTorch": "torch"},
    "MultipleModels-Vallina": {"PyTorch": "torch"},
    "MultipleModels-COM": {"PyTorch": "torch"},
    "MOBO-Vallina": {
        "PyTorch": "torch",
        "BoTorch": "botorch",
        "GPyTorch": "gpytorch",
    },
    "PCD": {"PyTorch": "torch"},
    "ParetoFlow": {"PyTorch": "torch"},
}


def project_python(*, require_exists: bool = True) -> Path:
    """Return the only interpreter used for real unified experiment runs."""

    if require_exists and not PROJECT_PYTHON.is_file():
        raise RuntimeError(
            f"Project virtual environment is missing: {PROJECT_PYTHON}.\n"
            "Create it from this repository with:\n"
            "  python3.11 scripts/setup_environment.py"
        )
    return PROJECT_PYTHON


def using_project_python(executable: str | Path | None = None) -> bool:
    candidate = Path(executable or sys.executable).resolve()
    if not PROJECT_PYTHON.exists():
        return False
    return candidate == PROJECT_PYTHON.resolve()


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _normalized_release(version: str) -> tuple[int | str, ...]:
    """Compare equivalent releases such as 1.2 and 1.2.0 without packaging."""

    release = version.split("+", 1)[0].split(".")
    while len(release) > 1 and release[-1] == "0":
        release.pop()
    return tuple(int(part) if part.isdigit() else part for part in release)


def _includes_molecule(problems: Iterable[str] | str | None) -> bool:
    if problems is None:
        # No CLI override means the configured full suite, which includes it.
        return True
    if isinstance(problems, str):
        problems = problems.split(",")
    return any(
        str(problem).strip().lower().replace("_", "-")
        in {"molecule", "molecule-exact-v0"}
        for problem in problems
    )


def environment_issues(
    methods: Iterable[str], problems: Iterable[str] | str | None = None
) -> list[str]:
    """Report missing local-runtime pieces for the selected methods."""

    issues = []
    if not using_project_python():
        issues.append(
            "the active interpreter is not this repository's .venv Python "
            f"({PROJECT_PYTHON})"
        )

    if sys.version_info[:2] != (3, 11):
        issues.append(
            f"Python 3.11 is required; active version is "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )

    required = dict(BASE_MODULES)
    if _includes_molecule(problems):
        required.update(MOLECULE_MODULES)
    for method in methods:
        required.update(METHOD_MODULES.get(method, {}))
    missing = sorted(label for label, module in required.items() if not _module_available(module))
    if missing:
        issues.append("missing packages: " + ", ".join(missing))

    mismatches = []
    for module in sorted(set(required.values())):
        distribution = MODULE_DISTRIBUTIONS.get(module)
        expected = PINNED_VERSIONS.get(distribution)
        if not distribution or not expected or not _module_available(module):
            continue
        try:
            installed = metadata.version(distribution).split("+", 1)[0]
        except metadata.PackageNotFoundError:
            continue
        if _normalized_release(installed) != _normalized_release(expected):
            mismatches.append(f"{distribution}=={installed} (expected {expected})")
    if mismatches:
        issues.append("version mismatches: " + ", ".join(mismatches))

    if "xgboost" in required.values() and _module_available("xgboost"):
        try:
            importlib.import_module("xgboost")
        except Exception as error:
            issues.append(
                "XGBoost native runtime failed to load: "
                f"{type(error).__name__}: {str(error).splitlines()[0]}"
            )

    if not (OFFLINE_MOO_ROOT / "off_moo_bench").is_dir():
        issues.append(
            "the repository submodule external/offline-moo is not initialized; run "
            "`git submodule update --init external/offline-moo`"
        )
    return issues


def format_environment_report(
    methods: Iterable[str], problems: Iterable[str] | str | None = None
) -> str:
    selected = tuple(methods)
    issues = environment_issues(selected, problems)
    if not issues:
        return (
            f"Environment OK: {PROJECT_PYTHON} | Python "
            f"{sys.version_info.major}.{sys.version_info.minor} | "
            f"{len(selected)} selected method(s)"
        )
    details = "\n".join(f"  - {issue}" for issue in issues)
    return (
        "Project environment is not ready:\n"
        f"{details}\n"
        "Install only from this repository:\n"
        "  python3.11 scripts/setup_environment.py"
    )
