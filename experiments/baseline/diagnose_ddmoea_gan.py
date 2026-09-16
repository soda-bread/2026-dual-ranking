#!/usr/bin/env python3
"""Compare paper-literal and small-data DDMOEA/GAN surrogate diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from pymoo.operators.sampling.lhs import LHS


HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.DL_MOBO_baseline.core import (  # noqa: E402
    ensure_import_paths,
    fit_neural_predictor,
    resolve_device,
    set_seed,
    test_mse,
)
from experiments.baseline.ddmoea_gan import (  # noqa: E402
    construct_surrogate_pool_with_gan,
    discriminator_scores,
    fit_discriminator_score_reference,
    surrogate_predict_with_ensemble,
    train_wgan_gp,
)
from src.official_pool import load_official_subset  # noqa: E402
from src.opt_problem import build_problem  # noqa: E402


PAPER_LITERAL = {
    "gan_hidden_layers": 1,
    "poly_ridge": False,
    "rbf_width": "paper",
    "rbf_center_cap": "none",
    # Algorithm 1 accumulates Ds, although the surrounding text says bagging.
    "accumulate_synthetic": True,
    # The paper specifies [0, 1] but not the normalization operation.
    "score_norm": "sigmoid",
}

FIELDS = (
    "problem",
    "variant",
    "dataset_source",
    "training_size",
    "decision_dimension",
    "MSEpre",
    "rbf_activation_mean",
    "rbf_sigma_mean",
    "rbf_center_count_mean",
    "rbf_train_size_min",
    "rbf_train_size_max",
    "poly_prediction_min",
    "poly_prediction_max",
    "observed_objective_min",
    "observed_objective_max",
    "poly_range_ratio",
    "critic_real_mean",
    "critic_synthetic_mean",
    "critic_real_minus_synthetic",
    "critic_separation_auc",
    "critic_score_reference",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="zdt1,re21")
    parser.add_argument("--training-size", type=int, default=100)
    parser.add_argument("--offline-seed", type=int, default=1)
    parser.add_argument("--test-seed", type=int, default=10001)
    parser.add_argument(
        "--dataset-source", choices=("lhs", "official_pool"), default=None
    )
    parser.add_argument("--gan-epochs", type=int, default=2000)
    parser.add_argument("--neural-epochs", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--config", type=Path, default=EXPERIMENTS_DIR / "config.yaml"
    )
    parser.add_argument(
        "--subset-cache-dir",
        type=Path,
        default=EXPERIMENTS_DIR / "data_subsets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENTS_DIR / "results" / "ddmoea_gan_diagnostics.csv",
    )
    args = parser.parse_args(argv)
    args.problems = tuple(
        item.strip().lower() for item in args.problems.split(",") if item.strip()
    )
    if not args.problems:
        parser.error("--problems must contain at least one problem")
    if args.training_size < 2 or args.gan_epochs < 1 or args.neural_epochs < 1:
        parser.error("training size and epoch counts must be positive")
    return args


def normalized_joint(x, y):
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    y_min, y_max = y.min(axis=0), y.max(axis=0)
    x_scaled = 2.0 * (x - x_min) / (x_max - x_min + 1e-12) - 1.0
    y_scaled = 2.0 * (y - y_min) / (y_max - y_min + 1e-12) - 1.0
    return np.hstack((x_scaled, y_scaled))


def load_diagnostic_data(problem_name, args):
    if args.dataset_source == "official_pool":
        return load_official_subset(
            cache_root=args.subset_cache_dir,
            problem_name=problem_name,
            sample_size=args.training_size,
            offline_seed=args.offline_seed,
            all_sample_sizes=(args.training_size,),
        )
    problem = build_problem(problem_name=problem_name)
    x_train = LHS()(problem, args.training_size, seed=args.offline_seed).get("X")
    y_train = problem.evaluate(x_train, return_values_of=["F"])
    x_test = LHS()(problem, args.training_size, seed=args.test_seed).get("X")
    y_test = problem.evaluate(x_test, return_values_of=["F"])
    return {
        "X_train": np.asarray(x_train),
        "y_train": np.asarray(y_train),
        "X_test": np.asarray(x_test),
        "y_test": np.asarray(y_test),
    }


def critic_diagnostics(generator, discriminator, device, joint_init, seed):
    set_seed(seed)
    with torch.no_grad():
        fake = generator(
            torch.randn(len(joint_init), generator.latent_dim, device=device)
        ).cpu().numpy()
    real_scores = discriminator_scores(joint_init, discriminator, device)
    fake_scores = discriminator_scores(fake, discriminator, device)
    labels = np.concatenate((np.ones(len(real_scores)), np.zeros(len(fake_scores))))
    scores = np.concatenate((real_scores, fake_scores))
    return {
        "critic_real_mean": float(real_scores.mean()),
        "critic_synthetic_mean": float(fake_scores.mean()),
        "critic_real_minus_synthetic": float(
            real_scores.mean() - fake_scores.mean()
        ),
        "critic_separation_auc": float(roc_auc_score(labels, scores)),
    }


def variant_row(
    problem,
    variant,
    options,
    data,
    generator,
    discriminator,
    device,
    common_diagnostics,
    seed,
):
    set_seed(seed)
    pools, _, diagnostics = construct_surrogate_pool_with_gan(
        X_init=data["X_train"],
        F_init=data["y_train"],
        generator=generator,
        discriminator=discriminator,
        device=device,
        n_models=data["X_train"].shape[1],
        select_ratio=0.2,
        poly_degree=2,
        poly_ridge=bool(options["poly_ridge"]),
        rbf_width=str(options["rbf_width"]),
        rbf_center_cap=str(options["rbf_center_cap"]),
        lambda_rbfn=1e-6,
        accumulate_synthetic=bool(options["accumulate_synthetic"]),
        return_diagnostics=True,
        verbose=False,
    )
    prediction = surrogate_predict_with_ensemble(data["X_test"], pools)
    observed_span = (
        diagnostics["observed_objective_max"]
        - diagnostics["observed_objective_min"]
    )
    polynomial_span = (
        diagnostics["poly_prediction_max"] - diagnostics["poly_prediction_min"]
    )
    row = {
        "problem": problem,
        "variant": variant,
        "dataset_source": data["dataset_source"],
        "training_size": len(data["X_train"]),
        "decision_dimension": data["X_train"].shape[1],
        "MSEpre": float(np.mean((prediction - data["y_test"]) ** 2)),
        **common_diagnostics,
    }
    for key, value in diagnostics.items():
        row[key] = (
            json.dumps(np.asarray(value).tolist())
            if np.asarray(value).ndim
            else value
        )
    row["poly_range_ratio"] = json.dumps(
        (polynomial_span / (observed_span + 1e-12)).tolist()
    )
    return row


def multiple_models_row(problem, data, seed, epochs, device, common_diagnostics):
    predictor = fit_neural_predictor(
        "MultipleModels-Vallina",
        data,
        {
            "hidden_sizes": [2048, 2048],
            "epochs": int(epochs),
            "batch_size": 128,
            "learning_rate": 0.001,
        },
        {},
        seed,
        device,
    )
    return {
        "problem": problem,
        "variant": "MultipleModels-Vallina",
        "dataset_source": data["dataset_source"],
        "training_size": len(data["X_train"]),
        "decision_dimension": data["X_train"].shape[1],
        "MSEpre": test_mse(predictor, data),
        **common_diagnostics,
    }


def main(argv=None):
    args = parse_args(argv)
    with args.config.open("r", encoding="utf-8") as handle:
        root_config = yaml.safe_load(handle) or {}
    if args.dataset_source is None:
        args.dataset_source = str(
            (root_config.get("sample_size_ablation") or {}).get(
                "dataset_source", "lhs"
            )
        )
    small_data = dict(root_config.get("ddmoea_gan") or {})
    ensure_import_paths()
    neural_device = resolve_device(args.device)
    rows = []

    for problem in args.problems:
        data = load_diagnostic_data(problem, args)
        data["dataset_source"] = args.dataset_source
        joint_init = normalized_joint(data["X_train"], data["y_train"])
        set_seed(args.offline_seed)
        generator, discriminator, gan_device = train_wgan_gp(
            joint_init=joint_init,
            d_dim=joint_init.shape[1],
            n_obj=data["y_train"].shape[1],
            n_epochs=args.gan_epochs,
            batch_size=64,
            gan_hidden_layers=1,
            lambda_gp=10.0,
            n_critic=5,
            lr=1e-4,
            verbose=False,
        )
        score_reference = fit_discriminator_score_reference(
            discriminator, joint_init, gan_device
        )
        common = critic_diagnostics(
            generator,
            discriminator,
            gan_device,
            joint_init,
            args.offline_seed,
        )
        common["critic_score_reference"] = json.dumps(score_reference)
        rows.append(
            variant_row(
                problem,
                "paper_literal",
                PAPER_LITERAL,
                data,
                generator,
                discriminator,
                gan_device,
                common,
                args.offline_seed,
            )
        )
        rows.append(
            variant_row(
                problem,
                "small_data",
                small_data,
                data,
                generator,
                discriminator,
                gan_device,
                common,
                args.offline_seed,
            )
        )
        rows.append(
            multiple_models_row(
                problem,
                data,
                args.offline_seed,
                args.neural_epochs,
                neural_device,
                common,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['problem']:6s} | {row['variant']:24s} | "
            f"MSEpre={float(row['MSEpre']):.6g}"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
