"""Ranking, calibration and decision-level metrics.

Calibration slope / intercept follow Van Calster et al. (2016) as used by
van den Goorbergh et al. (2022): the slope is the coefficient of a logistic
regression of y on logit(p); the intercept is fitted with the slope fixed at 1
(calibration-in-the-large).  Perfect calibration gives slope 1, intercept 0.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-size (quantile) bins: |mean(p) - mean(y)| weighted by bin size."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    total = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        total += len(b) / len(p) * abs(p[b].mean() - y[b].mean())
    return float(total)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict[str, list[float]]:
    order = np.argsort(p)
    out: dict[str, list[float]] = {"mean_pred": [], "frac_pos": [], "count": []}
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        out["mean_pred"].append(float(p[b].mean()))
        out["frac_pos"].append(float(y[b].mean()))
        out["count"].append(int(len(b)))
    return out


def _neg_log_lik(params: np.ndarray, z: np.ndarray, y: np.ndarray, fixed_slope: bool) -> float:
    a = params[0]
    b = 1.0 if fixed_slope else params[1]
    q = np.clip(sigmoid(a + b * z), EPS, 1 - EPS)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    z = logit(p)
    slope_fit = minimize(_neg_log_lik, x0=np.array([0.0, 1.0]), args=(z, y, False), method="BFGS")
    intercept_fit = minimize(_neg_log_lik, x0=np.array([0.0]), args=(z, y, True), method="BFGS")
    return float(slope_fit.x[1]), float(intercept_fit.x[0])


def choose_threshold(y_cal: np.ndarray, p_cal: np.ndarray) -> float:
    """Threshold that maximises F1 on the calibration split."""
    precision, recall, thresholds = precision_recall_curve(y_cal, p_cal)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], EPS)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")}


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    slope, intercept = calibration_slope_intercept(y, p)
    return {"brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
            "ece": expected_calibration_error(y, p),
            "cal_slope": slope, "cal_intercept": intercept}


def decision_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    return {"threshold": float(threshold),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0))}


class PlattScaler:
    """Sigmoid of a*logit(p)+b, fitted on the calibration split (Platt, 1999)."""

    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "PlattScaler":
        self.lr.fit(logit(p_cal)[:, None], y_cal)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(logit(p)[:, None])[:, 1]


def evaluate(y_cal: np.ndarray, p_cal: np.ndarray, y_test: np.ndarray, p_test: np.ndarray) -> dict[str, float]:
    """Everything the paper reports for one fitted model, raw and after Platt scaling."""
    out: dict[str, float] = {}
    out.update(ranking_metrics(y_test, p_test))
    out.update(calibration_metrics(y_test, p_test))
    out.update(decision_metrics(y_test, p_test, choose_threshold(y_cal, p_cal)))
    platt = PlattScaler().fit(p_cal, y_cal)
    p_platt = platt.transform(p_test)
    out.update({f"platt_{k}": v for k, v in calibration_metrics(y_test, p_platt).items()})
    out["platt_a"] = float(platt.lr.coef_[0, 0])
    out["platt_b"] = float(platt.lr.intercept_[0])
    return out
