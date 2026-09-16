"""Dependency-free method registry shared by all experiment entry points."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    dual_ranking: bool = False


METHOD_REGISTRY = {
    spec.name: spec
    for spec in (
        MethodSpec("GPR-RBF + NSGA-II", "gpr_rbf"),
        MethodSpec("GPR-RBF + NSGA-II + DR", "gpr_rbf", True),
        MethodSpec("GPR-Matern + NSGA-II", "gpr_matern"),
        MethodSpec("GPR-Matern + NSGA-II + DR", "gpr_matern", True),
        MethodSpec("QR + NSGA-II", "qr"),
        MethodSpec("QR + NSGA-II + DR", "qr", True),
        MethodSpec("BNN + NSGA-II", "bnn"),
        MethodSpec("BNN + NSGA-II + DR", "bnn", True),
        MethodSpec("XGBoost + NSGA-II", "xgboost"),
        MethodSpec("WeightedEnsemble L2 + NSGA-II", "ensemble"),
        MethodSpec("TGPR-MO", "tgpr_mo"),
        MethodSpec("DDMOEA-GAN", "ddmoea_gan"),
        MethodSpec("Prob-RVEA", "prob_rvea"),
        MethodSpec("Prob-MOEA/D", "prob_moead"),
        MethodSpec("TabPFN + NSGA-II", "tabpfn"),
    )
}

BASELINE_FAMILIES = {"prob_rvea", "prob_moead", "tgpr_mo", "ddmoea_gan"}

# These implementations remain registered and can be selected explicitly, but
# are temporarily excluded from the default experiment plan.
DEFAULT_DISABLED_METHODS = (
    "TabPFN + NSGA-II",
    "WeightedEnsemble L2 + NSGA-II",
    "XGBoost + NSGA-II",
)

__all__ = [
    "BASELINE_FAMILIES",
    "DEFAULT_DISABLED_METHODS",
    "METHOD_REGISTRY",
    "MethodSpec",
]
