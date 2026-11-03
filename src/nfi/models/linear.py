"""L2-regularised logistic regression baseline."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogRegModel:
    name = "logreg"

    def __init__(self, config: dict, *, n_features: int, class_weight: bool, seed: int, threads: int):
        self.config = config
        self.n_features = n_features
        self.class_weight = class_weight
        self.seed = seed
        self.model: LogisticRegression | None = None

    def fit(self, X_train, y_train, X_val, y_val, *, max_epochs: int, patience: int) -> "LogRegModel":
        self.model = LogisticRegression(
            C=self.config["C"], class_weight="balanced" if self.class_weight else None,
            max_iter=self.config["max_iter"], solver="lbfgs", random_state=self.seed)
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def size_info(self) -> dict:
        return {"n_parameters": int(self.n_features + 1)}

    def fit_info(self) -> dict:
        return {"n_iter": int(self.model.n_iter_[0])}
