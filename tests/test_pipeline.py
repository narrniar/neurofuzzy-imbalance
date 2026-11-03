"""Behavioural tests on synthetic data: metrics, calibration, leakage controls, models.

Run with:  .venv/bin/python -m pytest -q tests
"""
import numpy as np
import pandas as pd
import pytest

from nfi.data import Preprocessor, RawTable, Setting, make_split, subsample_positives
from nfi.metrics import (PlattScaler, calibration_slope_intercept, choose_threshold, evaluate,
                         expected_calibration_error)
from nfi.models import build
from nfi.models.mlp import hidden_for_budget, mlp_parameter_count, tsk_parameter_budget
from nfi.search import configurations


def synthetic(n: int = 4000, d: int = 6, rate: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    logits = 1.5 * X[:, 0] - X[:, 1] + np.log(rate / (1 - rate))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y


def table_from(X: np.ndarray, y: np.ndarray) -> RawTable:
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    frame["cat"] = np.where(X[:, 0] > 0, "a", "b")
    return RawTable("synthetic", frame, y, ["cat"], [f"f{i}" for i in range(X.shape[1])], None)


# ----------------------------------------------------------------------------- data
def test_subsample_hits_target_rate():
    _, y = synthetic(rate=0.3)
    keep = subsample_positives(y, 0.05, seed=1)
    assert abs(y[keep].mean() - 0.05) < 0.003
    assert (y[keep] == 0).sum() == (y == 0).sum()  # negatives untouched


def test_subsample_rejects_rate_above_natural():
    _, y = synthetic(rate=0.05)
    with pytest.raises(ValueError):
        subsample_positives(y, 0.5, seed=0)


def test_split_is_disjoint_and_stratified():
    X, y = synthetic()
    split = make_split(table_from(X, y), Setting("credit_default", "natural"), seed=0)
    parts = split.as_dict()
    all_rows = np.concatenate(list(parts.values()))
    assert len(np.unique(all_rows)) == len(all_rows) == len(y)
    assert len(parts["train"]) == pytest.approx(0.6 * len(y), abs=2)
    assert len(parts["test"]) == pytest.approx(0.2 * len(y), abs=2)
    rates = [y[v].mean() for v in parts.values()]
    assert max(rates) - min(rates) < 0.01


def test_time_split_never_trains_on_the_future():
    X, y = synthetic()
    table = table_from(X, y)
    table.order = np.random.default_rng(0).permutation(len(y)).astype(float)
    split = make_split(table, Setting("fraud", "natural"), seed=0)
    assert table.order[split.train].max() < table.order[split.val].min() < table.order[split.cal].min()
    assert table.order[split.cal].max() < table.order[split.test].min()


def test_preprocessor_fits_on_train_only():
    X, y = synthetic()
    table = table_from(X, y)
    split = make_split(table, Setting("credit_default", "natural"), seed=0)
    pre = Preprocessor(table).fit(split.train)
    Xtr = pre.transform(split.train)
    Xte = pre.transform(split.test)
    assert np.allclose(Xtr[:, : X.shape[1]].mean(axis=0), 0, atol=1e-5)
    assert not np.allclose(Xte[:, : X.shape[1]].mean(axis=0), 0, atol=1e-3)
    assert Xtr.shape[1] == X.shape[1] + 2  # two one-hot columns


# ----------------------------------------------------------------------------- metrics
def test_ece_zero_for_perfectly_calibrated_bins():
    p = np.repeat(np.linspace(0.05, 0.95, 10), 1000)
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=len(p)) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_calibration_slope_recovers_known_distortion():
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000)
    y = (rng.uniform(size=len(z)) < 1 / (1 + np.exp(-z))).astype(int)
    p_overconfident = 1 / (1 + np.exp(-2 * z))  # logits doubled -> slope 0.5
    slope, intercept = calibration_slope_intercept(y, p_overconfident)
    assert slope == pytest.approx(0.5, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.1)


def test_platt_fixes_overconfidence():
    rng = np.random.default_rng(1)
    z = rng.normal(size=20000)
    y = (rng.uniform(size=len(z)) < 1 / (1 + np.exp(-z))).astype(int)
    p = 1 / (1 + np.exp(-3 * z))
    fixed = PlattScaler().fit(p[:10000], y[:10000]).transform(p[10000:])
    assert expected_calibration_error(y[10000:], fixed) < expected_calibration_error(y[10000:], p[10000:])


def test_threshold_and_evaluate_keys():
    X, y = synthetic()
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - X[:, 1] - 2)))
    assert 0 < choose_threshold(y[:2000], p[:2000]) < 1
    out = evaluate(y[:2000], p[:2000], y[2000:], p[2000:])
    for k in ["pr_auc", "roc_auc", "brier", "ece", "cal_slope", "platt_ece", "f1", "threshold"]:
        assert k in out


# ----------------------------------------------------------------------------- models
@pytest.mark.parametrize("model", ["logreg", "lgbm", "mlp", "tsk"])
@pytest.mark.parametrize("class_weight", [False, True])
def test_every_model_beats_chance(model, class_weight):
    X, y = synthetic(rate=0.08)
    config = configurations(model, 1, seed=3)[0]
    clf = build(model, config, n_features=X.shape[1], class_weight=class_weight, seed=0, threads=1)
    clf.fit(X[:2500], y[:2500], X[2500:3000], y[2500:3000], max_epochs=30, patience=5)
    p = clf.predict_proba(X[3000:])
    assert p.shape == (1000,) and np.all((p >= 0) & (p <= 1))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[3000:], p) > 0.75
    assert clf.size_info()["n_parameters"] > 0


def test_tsk_fixed_membership_functions_do_not_move():
    X, y = synthetic()
    config = {**configurations("tsk", 1, seed=0)[0], "learn_mf": False, "n_rules": 4}
    clf = build("tsk", config, n_features=X.shape[1], class_weight=False, seed=0, threads=1)
    clf.fit(X[:3000], y[:3000], X[3000:], y[3000:], max_epochs=3, patience=3)
    assert not clf.net.centers.requires_grad
    assert clf.size_info()["n_trainable"] < clf.size_info()["n_parameters"]


def test_mlp_parameter_budget_matches_tsk():
    budget = tsk_parameter_budget(8, 30)
    hidden = hidden_for_budget(budget, 30, 1)
    assert abs(mlp_parameter_count(30, hidden, 1) - budget) / budget < 0.05


def test_configurations_are_reproducible():
    assert configurations("lgbm", 3, seed=7) == configurations("lgbm", 3, seed=7)
    assert configurations("tsk", 3, seed=7) != configurations("tsk", 3, seed=8)
