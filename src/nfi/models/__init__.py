"""Model registry.  Every model exposes fit / predict_proba / size_info / fit_info."""
from __future__ import annotations

from .linear import LogRegModel
from .mlp import MLPModel
from .trees import LightGBMModel
from .tsk import TSKModel

MODELS = {"logreg": LogRegModel, "lgbm": LightGBMModel, "mlp": MLPModel, "tsk": TSKModel}


def build(name: str, config: dict, *, n_features: int, class_weight: bool, seed: int, threads: int):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](config, n_features=n_features, class_weight=class_weight, seed=seed, threads=threads)
