"""Random hyperparameter configurations, one space per model.

Every model receives the same number of configurations per setting; the
configuration is selected by validation PR-AUC on the first seed.
"""
from __future__ import annotations

import numpy as np


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _choice(rng: np.random.Generator, options: list):
    return options[int(rng.integers(len(options)))]


def sample_tsk(rng: np.random.Generator) -> dict:
    return {
        "n_rules": _choice(rng, [4, 8, 16]),
        "learn_mf": True,
        "batch_norm": True,
        "sigma_scale": _choice(rng, [0.5, 1.0, 2.0]),
        "lr": _log_uniform(rng, 1e-3, 3e-2),
        "weight_decay": _choice(rng, [0.0, 1e-5, 1e-4, 1e-3]),
        "batch_size": _choice(rng, [256, 512, 1024]),
        "ur_weight": _choice(rng, [0.0, 0.1, 1.0]),
    }


def sample_mlp(rng: np.random.Generator) -> dict:
    return {
        "rules_equivalent": _choice(rng, [4, 8, 16]),
        "n_layers": _choice(rng, [1, 2]),
        "dropout": _choice(rng, [0.0, 0.1, 0.3]),
        "lr": _log_uniform(rng, 1e-3, 3e-2),
        "weight_decay": _choice(rng, [0.0, 1e-5, 1e-4, 1e-3]),
        "batch_size": _choice(rng, [256, 512, 1024]),
    }


def sample_lgbm(rng: np.random.Generator) -> dict:
    return {
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
        "learning_rate": _log_uniform(rng, 0.01, 0.3),
        "num_leaves": _choice(rng, [7, 15, 31, 63]),
        "min_child_samples": _choice(rng, [5, 20, 50, 100]),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "reg_lambda": _choice(rng, [0.0, 1.0, 10.0]),
    }


def sample_logreg(rng: np.random.Generator) -> dict:
    return {"C": _log_uniform(rng, 1e-3, 1e2), "max_iter": 2000}


SAMPLERS = {"tsk": sample_tsk, "mlp": sample_mlp, "lgbm": sample_lgbm, "logreg": sample_logreg}


def configurations(model: str, n_configs: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [SAMPLERS[model](rng) for _ in range(n_configs)]
