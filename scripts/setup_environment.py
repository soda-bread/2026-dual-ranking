#!/usr/bin/env python3
"""Create this repository's .venv and install its complete runtime."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_ROOT = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_ROOT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def repair_macos_xgboost_runtime() -> None:
    """Use scikit-learn's bundled OpenMP runtime inside the same .venv."""

    if sys.platform != "darwin":
        return
    site_packages = next(VENV_ROOT.glob("lib/python*/site-packages"), None)
    if site_packages is None:
        raise RuntimeError("could not locate .venv site-packages")
    source = site_packages / "sklearn" / ".dylibs" / "libomp.dylib"
    xgboost_library = site_packages / "xgboost" / "lib" / "libxgboost.dylib"
    destination = xgboost_library.parent / "libomp.dylib"
    if not source.is_file() or not xgboost_library.is_file():
        raise RuntimeError("the installed scikit-learn/xgboost wheels are incomplete")
    shutil.copy2(source, destination)
    rpaths = subprocess.check_output(
        ["otool", "-l", str(xgboost_library)], text=True
    )
    if "path @loader_path " not in rpaths:
        subprocess.check_call(
            ["install_name_tool", "-add_rpath", "@loader_path", str(xgboost_library)]
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-submodule",
        action="store_true",
        help="do not initialize the repository-owned offline-moo submodule",
    )
    args = parser.parse_args(argv)

    if sys.version_info[:2] != (3, 11):
        parser.error(
            f"run this script with Python 3.11, not "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )

    if not args.skip_submodule:
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "external/offline-moo"],
            cwd=REPO_ROOT,
        )

    if not VENV_PYTHON.is_file():
        venv.EnvBuilder(with_pip=True).create(VENV_ROOT)

    subprocess.check_call(
        [
            str(VENV_PYTHON),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        cwd=REPO_ROOT,
    )
    subprocess.check_call(
        [str(VENV_PYTHON), "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=REPO_ROOT,
    )
    repair_macos_xgboost_runtime()
    subprocess.check_call(
        [
            str(VENV_PYTHON),
            "experiments/run_all_methods.py",
            "--check-environment",
        ],
        cwd=REPO_ROOT,
    )
    print(f"Project environment is ready: {VENV_PYTHON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
