import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
FORBIDDEN_TRAINING_PATTERNS = re.compile(
    r"k[ _-]?fold|cross[ _-]?validation|cross_validate|cross_val|"
    r"\bX_val\b|\by_val\b|validation_(?:size|seed|fraction)|"
    r"val_split|early[ _-]?stop|conformal_upper",
    flags=re.IGNORECASE,
)


def _notebook_code(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


class ExperimentDataProtocolTests(unittest.TestCase):
    def test_all_exp_entrypoints_have_no_kfold_or_validation_split(self):
        python_entries = sorted(EXPERIMENTS_DIR.glob("Exp[0-9]*_*.py"))
        notebook_entries = sorted(EXPERIMENTS_DIR.glob("Exp[0-9]*_*.ipynb"))
        self.assertEqual(len(python_entries), 8)
        self.assertEqual(len(notebook_entries), 8)

        sources = {
            path: path.read_text(encoding="utf-8") for path in python_entries
        }
        sources.update({path: _notebook_code(path) for path in notebook_entries})
        for path, source in sources.items():
            self.assertIsNone(
                FORBIDDEN_TRAINING_PATTERNS.search(source),
                msg=f"Forbidden split/training pattern found in {path}",
            )

    def test_shared_experiment_training_paths_have_no_split_logic(self):
        paths = (
            REPO_ROOT / "src" / "data.py",
            REPO_ROOT / "src" / "models.py",
            REPO_ROOT / "src" / "official_pool.py",
            EXPERIMENTS_DIR / "sample_size_common.py",
            EXPERIMENTS_DIR / "baseline" / "batch_experiments.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIsNone(
                FORBIDDEN_TRAINING_PATTERNS.search(source),
                msg=f"Forbidden split/training pattern found in {path}",
            )


if __name__ == "__main__":
    unittest.main()
