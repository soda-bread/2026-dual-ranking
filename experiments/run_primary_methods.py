#!/usr/bin/env python3
"""Run the normal, DR, and EBU-DR GPR/QR/BNN primary methods."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_all_methods import (  # noqa: E402
    DR_METHODS,
    EBU_DR_METHODS,
    NORMAL_METHODS,
    PRIMARY_EXPERIMENT_METHODS,
    fixed_group_main,
)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list-methods" in argv:
        for category, methods in (
            ("normal", NORMAL_METHODS),
            ("dr", DR_METHODS),
            ("ebu_dr", EBU_DR_METHODS),
        ):
            print(f"{category} ({len(methods)}):")
            for method in methods:
                print(f"  {method}")
        return 0
    return fixed_group_main(
        "primary methods",
        PRIMARY_EXPERIMENT_METHODS,
        argv,
        default_output_dir=REPO_ROOT / "experiments" / "results_primary_methods",
    )


if __name__ == "__main__":
    raise SystemExit(main())
