"""Small MLP whose parameter count is matched to a TSK model with the same rule count.

A TSK model with R rules on D features has R*(2D) membership + R*(D+1) consequent
parameters.  `hidden_for_budget` picks the hidden width of a one- or two-layer
MLP so that its parameter count is as close as possible to that budget.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .torch_train import TorchClassifier, base_rate_log_odds, count_parameters, predict_logits, train


def tsk_parameter_budget(n_rules: int, n_features: int) -> int:
    return n_rules * (2 * n_features) + n_rules * (n_features + 1)


def mlp_parameter_count(n_features: int, hidden: int, n_layers: int) -> int:
    count = n_features * hidden + hidden
    for _ in range(n_layers - 1):
        count += hidden * hidden + hidden
    return count + hidden + 1


def hidden_for_budget(budget: int, n_features: int, n_layers: int) -> int:
    best, best_gap = 1, float("inf")
    for h in range(1, 1024):
        gap = abs(mlp_parameter_count(n_features, h, n_layers) - budget)
        if gap < best_gap:
            best, best_gap = h, gap
        if mlp_parameter_count(n_features, h, n_layers) > budget:
            break
    return best


class MLP(TorchClassifier):
    def __init__(self, n_features: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).squeeze(-1)


class MLPModel:
    name = "mlp"

    def __init__(self, config: dict, *, n_features: int, class_weight: bool, seed: int, threads: int):
        self.config = config
        self.n_features = n_features
        self.class_weight = class_weight
        self.seed = seed
        self.threads = threads
        budget = tsk_parameter_budget(config["rules_equivalent"], n_features)
        self.hidden = hidden_for_budget(budget, n_features, config["n_layers"])
        self.net: MLP | None = None
        self.info: dict = {}

    def fit(self, X_train, y_train, X_val, y_val, *, max_epochs: int, patience: int) -> "MLPModel":
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        self.net = MLP(self.n_features, self.hidden, self.config["n_layers"], self.config["dropout"])
        with torch.no_grad():
            self.net.body[-1].bias.fill_(base_rate_log_odds(y_train, self.class_weight))
        pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1)) if self.class_weight else None
        self.info = train(
            self.net, X_train, y_train, X_val, y_val,
            lr=self.config["lr"], weight_decay=self.config["weight_decay"], batch_size=self.config["batch_size"],
            max_epochs=max_epochs, patience=patience, pos_weight=pos_weight, reg_weight=0.0, seed=self.seed)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-predict_logits(self.net, X).astype(np.float64)))

    def size_info(self) -> dict:
        return {"hidden": self.hidden, "n_layers": self.config["n_layers"], **count_parameters(self.net)}

    def fit_info(self) -> dict:
        return self.info
