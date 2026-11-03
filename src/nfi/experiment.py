"""Experiment runner: random search -> final seeds -> TSK ablations, fully resumable.

Every model fit is one record in results/runs/<experiment>.jsonl with its run id,
timestamp, setting, seed, configuration, status, metrics, timings and the path of
its saved predictions.  A record's key is a hash of everything that determines
the fit, so re-running the same configuration file skips finished work.
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .data import ROOT, Prepared, Setting, prepare
from .metrics import evaluate, ranking_metrics
from .models import build
from .search import configurations
from .seeds import set_seed

ABLATION_RULES = [4, 8, 16]


@dataclass
class Experiment:
    name: str
    seeds: list[int]
    n_configs: int
    max_epochs: int
    patience: int
    settings: list[str]
    models: list[str]
    weighting: list[bool]
    ablations: bool = True
    threads: int = 4
    results_dir: str = "results"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "Experiment":
        raw = yaml.safe_load(Path(path).read_text())
        known = {k: raw.pop(k) for k in list(raw) if k in Experiment.__dataclass_fields__}
        return Experiment(**known, extra=raw)

    @property
    def out(self) -> Path:
        return ROOT / self.results_dir

    @property
    def runs_dir(self) -> Path:
        return self.out / "runs"

    def runs_file(self, slug: str = "") -> Path:
        """One log per process: parallel runs over setting subsets write to different files."""
        return self.runs_dir / (f"{self.name}.{slug}.jsonl" if slug else f"{self.name}.jsonl")

    def all_runs_files(self) -> list[Path]:
        return sorted(self.runs_dir.glob(f"{self.name}.jsonl")) + sorted(self.runs_dir.glob(f"{self.name}.*.jsonl"))

    def all_ablation_files(self) -> list[Path]:
        return sorted(self.runs_dir.glob(f"{self.name}*.ablations.json"))


def run_key(setting: Setting, model: str, class_weight: bool, features: str, config: dict, seed: int,
            evaluate_test: bool) -> str:
    payload = json.dumps([setting.key, model, class_weight, features, config, seed, evaluate_test], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


class RunLog:
    """Finished records from every log of the experiment; new records go to `path`."""

    def __init__(self, path: Path, read_from: list[Path] | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        for source in read_from or [path]:
            if not source.exists():
                continue
            for line in source.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec["status"] == "ok":
                        self.records[rec["key"]] = rec

    def append(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["status"] == "ok":
            self.records[rec["key"]] = rec


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # bytes on macOS


class Runner:
    def __init__(self, exp: Experiment, settings: list[str] | None = None, models: list[str] | None = None,
                 threads: int | None = None):
        self.exp = exp
        if threads:
            self.exp.threads = threads
        self.settings = [Setting.parse(s) for s in (settings or exp.settings)]
        self.models = models or exp.models
        self.slug = "_".join(s.key.replace(":", "-") for s in self.settings) if settings else ""
        self.log = RunLog(exp.runs_file(self.slug), exp.all_runs_files())
        self.preds_dir = exp.out / "preds"
        self.preds_dir.mkdir(parents=True, exist_ok=True)
        self.split_dir = exp.out / "splits"
        self._data: dict[tuple, Prepared] = {}
        self.ablation_index: dict = self._load_ablation_index()

    # ------------------------------------------------------------------ data
    def data(self, setting: Setting, seed: int, features: str) -> Prepared:
        key = (setting.key, seed, features)
        if key not in self._data:
            if len(self._data) > 12:  # keep memory bounded on the fraud table
                self._data.clear()
            self._data[key] = prepare(setting, seed, features, self.split_dir)
        return self._data[key]

    # ------------------------------------------------------------------ one fit
    def fit_one(self, phase: str, setting: Setting, model: str, class_weight: bool, features: str,
                config: dict, config_idx: int, seed: int, evaluate_test: bool) -> dict:
        key = run_key(setting, model, class_weight, features, config, seed, evaluate_test)
        if key in self.log.records:
            return self.log.records[key]
        data = self.data(setting, seed, features)
        rec = {
            "run_id": uuid.uuid4().hex[:12], "key": key, "time": datetime.now(timezone.utc).isoformat(),
            "experiment": self.exp.name, "phase": phase, "setting": setting.key, "dataset": setting.dataset,
            "rate": setting.rate, "model": model, "class_weight": class_weight, "features": features,
            "config": config, "config_idx": config_idx, "seed": seed, "n_features": data.n_features,
            "split_sizes": {k: int(len(v)) for k, v in data.rows.items()},
            "positives": {k: int(v.sum()) for k, v in data.y.items()},
            "duplicates_train_test": data.duplicates_train_test,
        }
        try:
            rec.update(self._fit_and_score(data, model, class_weight, config, seed, evaluate_test))
            rec["status"] = "ok"
        except Exception as e:  # noqa: BLE001 - a 40 hour job must survive one bad fit
            rec.update({"status": "failed", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
            print(f"  FAILED {model} {setting.key} cw={class_weight} seed={seed}: {rec['error']}", file=sys.stderr)
        self.log.append(rec)
        self._print(rec)
        return rec

    def _fit_and_score(self, data: Prepared, model: str, class_weight: bool, config: dict, seed: int,
                       evaluate_test: bool) -> dict:
        set_seed(seed)
        clf = build(model, config, n_features=data.n_features, class_weight=class_weight, seed=seed,
                    threads=self.exp.threads)
        t0 = time.perf_counter()
        clf.fit(data.X["train"], data.y["train"], data.X["val"], data.y["val"],
                max_epochs=self.exp.max_epochs, patience=self.exp.patience)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        p_val = clf.predict_proba(data.X["val"])
        predict_s = (time.perf_counter() - t1) / max(len(p_val), 1) * 1000.0  # ms per 1000 rows
        out = {"fit_seconds": fit_s, "predict_ms_per_1k": predict_s, "peak_rss_mb": _peak_rss_mb(),
               "size": clf.size_info(), "fit_info": clf.fit_info(),
               "val": ranking_metrics(data.y["val"], p_val)}
        if evaluate_test:
            p_cal, p_test = clf.predict_proba(data.X["cal"]), clf.predict_proba(data.X["test"])
            out["test"] = evaluate(data.y["cal"], p_cal, data.y["test"], p_test)
            path = self.preds_dir / f"{run_key(data.setting, model, class_weight, data.features, config, seed, True)}.npz"
            np.savez_compressed(path, p_val=p_val, p_cal=p_cal, p_test=p_test,
                                y_cal=data.y["cal"], y_test=data.y["test"], rows_test=data.rows["test"])
            out["preds_path"] = str(path.relative_to(ROOT))
        return out

    @staticmethod
    def _print(rec: dict) -> None:
        if rec["status"] != "ok":
            return
        score = rec["test"]["pr_auc"] if "test" in rec else rec["val"]["pr_auc"]
        where = "test" if "test" in rec else "val"
        print(f"[{rec['phase']:<8}] {rec['setting']:<22} {rec['model']:<6} cw={int(rec['class_weight'])} "
              f"feat={rec['features']:<5} seed={rec['seed']} {where}_pr_auc={score:.4f} ({rec['fit_seconds']:.1f}s)",
              flush=True)

    # ------------------------------------------------------------------ phases
    def search(self, setting: Setting, model: str, class_weight: bool, features: str = "all") -> tuple[dict, int]:
        config_seed = int(hashlib.sha1(f"{setting.key}|{model}".encode()).hexdigest(), 16) % (2 ** 31)
        configs = configurations(model, self.exp.n_configs, config_seed)
        best, best_idx, best_score = None, -1, -np.inf
        for idx, config in enumerate(configs):
            rec = self.fit_one("search", setting, model, class_weight, features, config, idx, self.exp.seeds[0], False)
            if rec["status"] == "ok" and rec["val"]["pr_auc"] > best_score:
                best, best_idx, best_score = config, idx, rec["val"]["pr_auc"]
        if best is None:
            raise RuntimeError(f"every configuration failed for {model} on {setting.key}")
        return best, best_idx

    def final(self, phase: str, setting: Setting, model: str, class_weight: bool, config: dict, config_idx: int,
              features: str = "all") -> list[str]:
        keys = []
        for seed in self.exp.seeds:
            rec = self.fit_one(phase, setting, model, class_weight, features, config, config_idx, seed, True)
            keys.append(rec["key"])
        return keys

    def run(self) -> None:
        for setting in self.settings:
            best: dict[tuple[str, bool], tuple[dict, int]] = {}
            for model in self.models:
                for class_weight in self.exp.weighting:
                    config, idx = self.search(setting, model, class_weight)
                    best[(model, class_weight)] = (config, idx)
                    self.final("final", setting, model, class_weight, config, idx)
            if self.exp.ablations:
                self.ablate(setting, best)
        print(f"done: {self.log.path}")

    # ------------------------------------------------------------------ ablations (SQ2 + feature subset)
    def ablate(self, setting: Setting, best: dict[tuple[str, bool], tuple[dict, int]]) -> None:
        index = self.ablation_index.setdefault(setting.key, {})
        if ("tsk", False) in best:
            base, idx = best[("tsk", False)]
            variants = {"base": base, "mf=fixed": {**base, "learn_mf": False}}
            for rules in ABLATION_RULES:
                variants[f"rules={rules}"] = {**base, "n_rules": rules}
            variants["ur=off"] = {**base, "ur_weight": 0.0}
            variants["ur=on"] = {**base, "ur_weight": base["ur_weight"] if base["ur_weight"] > 0 else 1.0}
            for name, config in variants.items():
                index[f"tsk|{name}"] = self.final("ablation", setting, "tsk", False, config, idx)
        for (model, class_weight), (config, idx) in best.items():
            if class_weight:
                continue
            index[f"{model}|features=top10"] = self.final("ablation", setting, model, False, config, idx, "top10")
        self._save_ablation_index()

    def _ablation_index_path(self) -> Path:
        return self.log.path.with_suffix("").with_name(self.log.path.stem + ".ablations.json")

    def _load_ablation_index(self) -> dict:
        path = self._ablation_index_path()
        return json.loads(path.read_text()) if path.exists() else {}

    def _save_ablation_index(self) -> None:
        self._ablation_index_path().write_text(json.dumps(self.ablation_index, indent=2))
