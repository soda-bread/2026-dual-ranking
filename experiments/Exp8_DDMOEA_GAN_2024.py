# Auto-generated runner for DDMOEA_GAN.
# Run with: python Exp8_DDMOEA_GAN_2024.py

from pathlib import Path as _ExperimentsPath
import atexit as _experiments_atexit
import sys as _experiments_sys
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*datetime\.datetime\.utcnow\(\) is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"paramz(\..*)?")
warnings.filterwarnings("ignore", message=r"Failed to config .* module.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Gym has been unmaintained.*", category=UserWarning)

EXPERIMENTS_DIR = _ExperimentsPath(__file__).resolve().parent
EXPERIMENTS_LOG_DIR = EXPERIMENTS_DIR / "logs"
EXPERIMENTS_RESULTS_DIR = EXPERIMENTS_DIR / "results"
EXPERIMENTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_EXPERIMENTS_METHOD_NAME = "DDMOEA_GAN"
_EXPERIMENTS_LOG_PATH = EXPERIMENTS_LOG_DIR / (_EXPERIMENTS_METHOD_NAME + ".log")
_EXPERIMENTS_LOG_FILE = open(_EXPERIMENTS_LOG_PATH, "a", encoding="utf-8", buffering=1)
_EXPERIMENTS_STDOUT = _experiments_sys.stdout
_EXPERIMENTS_STDERR = _experiments_sys.stderr


class _ExperimentsTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


_experiments_sys.stdout = _ExperimentsTee(_EXPERIMENTS_STDOUT, _EXPERIMENTS_LOG_FILE)
_experiments_sys.stderr = _ExperimentsTee(_EXPERIMENTS_STDERR, _EXPERIMENTS_LOG_FILE)


def _close_experiments_log():
    _experiments_sys.stdout = _EXPERIMENTS_STDOUT
    _experiments_sys.stderr = _EXPERIMENTS_STDERR
    _EXPERIMENTS_LOG_FILE.flush()
    _EXPERIMENTS_LOG_FILE.close()


_experiments_atexit.register(_close_experiments_log)
print(f"[experiments] logging to: {_EXPERIMENTS_LOG_PATH}")
print(f"[experiments] results dir: {EXPERIMENTS_RESULTS_DIR}")

import importlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True

if "imp" not in sys.modules and importlib.util.find_spec("imp") is None:
    sys.modules["imp"] = types.ModuleType("imp")

DEPENDENCIES = {
    "pymoo": "pymoo==0.6.1.6",
    "pyDOE2": "pyDOE2",
    "GPy": "GPy",
    "yaml": "pyyaml",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "diversipy": "diversipy",
    "optproblems": "optproblems",
    "graphviz": "graphviz",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
}

if _EXPERIMENTS_METHOD_NAME == "DDMOEA_GAN":
    DEPENDENCIES.update({"torch": "torch"})

for import_name, pip_name in DEPENDENCIES.items():
    try:
        importlib.import_module(import_name)
        print(f"{import_name} is available.")
    except ImportError:
        print(f"Installing {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pip_name])

repo_candidates = [
    Path.cwd().resolve(),
    Path.cwd().resolve().parent,
    EXPERIMENTS_DIR.parent,
    Path('/content/drive/MyDrive/2026 Real-wrold problem'),
    Path('/rds/projects/w/wangsu-building-automation/Huanbo/2026_real_world_problem'),
]
for repo_root in repo_candidates:
    if (repo_root / 'src').exists() and (repo_root / 'experiments' / 'baseline' / 'batch_experiments.py').exists():
        repo_root_string = str(repo_root)
        while repo_root_string in sys.path:
            sys.path.remove(repo_root_string)
        sys.path.insert(0, repo_root_string)
        baseline_root = repo_root / 'experiments'
        while str(baseline_root) in sys.path:
            sys.path.remove(str(baseline_root))
        sys.path.insert(0, str(baseline_root))
        print(f"Using repository root: {repo_root}")
        break
else:
    raise FileNotFoundError('Could not locate current repository root with src and experiments/baseline.')

importlib.invalidate_caches()
sys.modules.pop('baseline.batch_experiments', None)
sys.modules.pop('baseline', None)
import baseline.batch_experiments as batch_experiments

print(f"Loaded baseline runner: {batch_experiments.__file__}")
all_results = batch_experiments.run_ddmoea_gan_suite(config_path=EXPERIMENTS_DIR / 'config.yaml')
