"""Shared mini-batch training loop for the PyTorch models (TSK and MLP).

Cross-entropy (BCE on logits) with optional positive-class weighting, Adam,
early stopping on validation PR-AUC, restore of the best epoch.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn


class TorchClassifier(nn.Module):
    """Base for models trained by `train`.  Subclasses may override `regularizer`."""

    def regularizer(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)


def predict_logits(model: nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            out.append(model(torch.from_numpy(X[start:start + batch_size])).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train(model: TorchClassifier, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
          *, lr: float, weight_decay: float, batch_size: int, max_epochs: int, patience: int,
          pos_weight: float | None, reg_weight: float, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train.astype(np.float32))
    pw = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score, best_state, best_epoch, bad_epochs = -np.inf, copy.deepcopy(model.state_dict()), 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue  # BatchNorm needs more than one sample
            optimizer.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            if reg_weight > 0:
                loss = loss + reg_weight * model.regularizer()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(idx)
        score = float(average_precision_score(y_val, predict_logits(model, X_val)))
        history.append({"epoch": epoch, "train_loss": epoch_loss / len(Xt), "val_pr_auc": score})
        if score > best_score + 1e-6:
            best_score, best_state, best_epoch, bad_epochs = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"epochs_run": len(history), "best_epoch": best_epoch, "best_val_pr_auc": best_score}


def base_rate_log_odds(y: np.ndarray, class_weight: bool) -> float:
    """Log-odds of the positive class; with class weighting the effective prior is 1:1."""
    if class_weight:
        return 0.0
    rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    return float(np.log(rate / (1 - rate)))


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"n_parameters": int(total), "n_trainable": int(trainable)}
