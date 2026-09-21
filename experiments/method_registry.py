"""Dependency-free method registry shared by all experiment entry points."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    category: str = "normal"


METHOD_REGISTRY = {
    spec.name: spec
    for spec in (
        MethodSpec("GPR-RBF + NSGA-II", "gpr_rbf", "normal"),
        MethodSpec("GPR-Matern + NSGA-II", "gpr_matern", "normal"),
        MethodSpec("QR + NSGA-II", "qr", "normal"),
        MethodSpec("BNN + NSGA-II", "bnn", "normal"),
        MethodSpec("GPR-RBF + NSGA-II + DR", "gpr_rbf", "dr"),
        MethodSpec("GPR-Matern + NSGA-II + DR", "gpr_matern", "dr"),
        MethodSpec("QR + NSGA-II + DR", "qr", "dr"),
        MethodSpec("BNN + NSGA-II + DR", "bnn", "dr"),
        MethodSpec("GPR-RBF + NSGA-II + EBU-DR", "gpr_rbf", "ebu_dr"),
        MethodSpec("GPR-Matern + NSGA-II + EBU-DR", "gpr_matern", "ebu_dr"),
        MethodSpec("QR + NSGA-II + EBU-DR", "qr", "ebu_dr"),
        MethodSpec("BNN + NSGA-II + EBU-DR", "bnn", "ebu_dr"),
        MethodSpec("XGBoost + NSGA-II", "xgboost", "hidden"),
        MethodSpec("WeightedEnsemble L2 + NSGA-II", "ensemble", "hidden"),
        MethodSpec("TGPR-MO", "tgpr_mo", "baseline"),
        MethodSpec("DDMOEA-GAN", "ddmoea_gan", "baseline"),
        MethodSpec("Prob-RVEA", "prob_rvea", "baseline"),
        MethodSpec("Prob-MOEA/D", "prob_moead", "baseline"),
        MethodSpec("TabPFN + NSGA-II", "tabpfn", "hidden"),
    )
}

BASELINE_FAMILIES = {"prob_rvea", "prob_moead", "tgpr_mo", "ddmoea_gan"}

NORMAL_METHODS = tuple(
    spec.name for spec in METHOD_REGISTRY.values() if spec.category == "normal"
)
DR_METHODS = tuple(
    spec.name for spec in METHOD_REGISTRY.values() if spec.category == "dr"
)
EBU_DR_METHODS = tuple(
    spec.name for spec in METHOD_REGISTRY.values() if spec.category == "ebu_dr"
)

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
    "DR_METHODS",
    "EBU_DR_METHODS",
    "METHOD_REGISTRY",
    "MethodSpec",
    "NORMAL_METHODS",
]
