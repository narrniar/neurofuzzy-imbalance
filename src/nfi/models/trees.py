"""LightGBM baseline with early stopping on validation PR-AUC."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np


class LightGBMModel:
    name = "lgbm"

    def __init__(self, config: dict, *, n_features: int, class_weight: bool, seed: int, threads: int):
        self.config = config
        self.class_weight = class_weight
        self.seed = seed
        self.threads = threads
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, X_train, y_train, X_val, y_val, *, max_epochs: int, patience: int) -> "LightGBMModel":
        scale = float((y_train == 0).sum() / max((y_train == 1).sum(), 1)) if self.class_weight else 1.0
        self.model = lgb.LGBMClassifier(
            n_estimators=self.config["n_estimators"], learning_rate=self.config["learning_rate"],
            num_leaves=self.config["num_leaves"], min_child_samples=self.config["min_child_samples"],
            subsample=self.config["subsample"], subsample_freq=1, colsample_bytree=self.config["colsample_bytree"],
            reg_lambda=self.config["reg_lambda"], scale_pos_weight=scale,
            random_state=self.seed, n_jobs=self.threads, verbose=-1)
        self.model.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, eval_metric="average_precision",
                       callbacks=[lgb.early_stopping(self.config["early_stopping_rounds"], verbose=False)])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def size_info(self) -> dict:
        booster = self.model.booster_
        n_trees = booster.num_trees()
        leaves = sum(t["num_leaves"] for t in booster.dump_model()["tree_info"])
        return {"n_trees": int(n_trees), "n_leaves": int(leaves), "n_parameters": int(leaves)}

    def fit_info(self) -> dict:
        return {"best_iteration": int(self.model.best_iteration_ or self.model.n_estimators)}
