"""First-order TSK fuzzy classifier with Gaussian membership functions (HTSK variant).

Rule r:  IF x_1 is A_{r,1} AND ... AND x_D is A_{r,D}  THEN  y_r = w_r . x + b_r
Firing level (HTSK, Cui et al. 2021): the log-firing is the *mean* over dimensions
of -(x_d - c_{r,d})^2 / (2 sigma_{r,d}^2) instead of the sum, which keeps the
normalised firing levels from saturating when D is large.
Output logit = sum_r fbar_r * y_r with fbar = softmax_r(log firing).

Training options from Cui et al. (2020): mini-batch gradient descent, batch
normalisation of the rule consequents and uniform regularisation (UR), which
pushes every rule to fire equally on average over a batch.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from .torch_train import TorchClassifier, base_rate_log_odds, count_parameters, predict_logits, train


class HTSK(TorchClassifier):
    def __init__(self, centers: np.ndarray, sigmas: np.ndarray, *, learn_mf: bool, batch_norm: bool):
        super().__init__()
        n_rules, n_features = centers.shape
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32), requires_grad=learn_mf)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigmas, dtype=torch.float32)), requires_grad=learn_mf)
        self.consequent = nn.Linear(n_features, n_rules)
        self.bn = nn.BatchNorm1d(n_rules) if batch_norm else nn.Identity()
        self.last_firing: torch.Tensor | None = None

    def init_output_bias(self, log_odds: float) -> None:
        """Start every rule at the base-rate log-odds so the first prediction is the prior,
        not 0.5 (the same prior-bias initialisation is applied to the MLP)."""
        with torch.no_grad():
            if isinstance(self.bn, nn.BatchNorm1d):
                self.bn.bias.fill_(log_odds)
            else:
                self.consequent.bias.fill_(log_odds)

    def firing(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        d2 = (x[:, None, :] - self.centers[None, :, :]) ** 2 / (2 * sigma[None, :, :] ** 2)
        log_firing = -d2.mean(dim=-1)  # HTSK: mean over D, not sum
        return torch.softmax(log_firing, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fbar = self.firing(x)
        self.last_firing = fbar
        rule_outputs = self.bn(self.consequent(x))
        return (fbar * rule_outputs).sum(dim=1)

    def regularizer(self) -> torch.Tensor:
        """Uniform regularisation: mean firing of each rule over the batch should be 1/R."""
        if self.last_firing is None:
            return torch.zeros(())
        n_rules = self.last_firing.shape[1]
        return ((self.last_firing.mean(dim=0) - 1.0 / n_rules) ** 2).sum()


class TSKModel:
    name = "tsk"

    def __init__(self, config: dict, *, n_features: int, class_weight: bool, seed: int, threads: int):
        self.config = config
        self.n_features = n_features
        self.class_weight = class_weight
        self.seed = seed
        self.threads = threads
        self.net: HTSK | None = None
        self.info: dict = {}

    def _init_rules(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_rules = self.config["n_rules"]
        km = KMeans(n_clusters=n_rules, n_init=1, random_state=self.seed).fit(X)
        centers = km.cluster_centers_
        std = X.std(axis=0) + 1e-3
        sigmas = np.tile(std * self.config["sigma_scale"], (n_rules, 1))
        return centers, sigmas

    def fit(self, X_train, y_train, X_val, y_val, *, max_epochs: int, patience: int) -> "TSKModel":
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        centers, sigmas = self._init_rules(X_train)
        self.net = HTSK(centers, sigmas, learn_mf=self.config["learn_mf"], batch_norm=self.config["batch_norm"])
        self.net.init_output_bias(base_rate_log_odds(y_train, self.class_weight))
        pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1)) if self.class_weight else None
        self.info = train(
            self.net, X_train, y_train, X_val, y_val,
            lr=self.config["lr"], weight_decay=self.config["weight_decay"], batch_size=self.config["batch_size"],
            max_epochs=max_epochs, patience=patience, pos_weight=pos_weight,
            reg_weight=self.config["ur_weight"], seed=self.seed)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-predict_logits(self.net, X).astype(np.float64)))

    def size_info(self) -> dict:
        n_rules = self.config["n_rules"]
        return {"n_rules": n_rules, "antecedents_per_rule": self.n_features,
                "consequent_parameters": n_rules * (self.n_features + 1),
                "membership_parameters": 2 * n_rules * self.n_features, **count_parameters(self.net)}

    def fit_info(self) -> dict:
        return self.info
