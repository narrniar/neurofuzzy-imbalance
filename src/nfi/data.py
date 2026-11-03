"""Dataset loading, imbalance subsampling, splitting and leakage-safe preprocessing.

Every fitted preprocessing step (one-hot vocabulary, scaler, feature ranking, k-means)
is fitted on the training split only.  Split indices always refer to row positions
of the raw table so that any prediction can be traced back to its source row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

DATASETS = {
    "credit_default": {"file": "default_of_credit_card_clients.xls", "time_split": False},
    "bank_marketing": {"file": "bank-additional-full.csv", "time_split": False},
    "fraud": {"file": "creditcard.arff", "time_split": True},
}


@dataclass
class RawTable:
    name: str
    frame: pd.DataFrame          # features only, raw row order, RangeIndex
    y: np.ndarray                # 0/1
    categorical: list[str]
    numeric: list[str]
    order: np.ndarray | None     # time ordering key for time-based split, else None
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)  # subgroup columns (sex, age)


# ----------------------------------------------------------------------------- loading
def load_credit_default() -> RawTable:
    df = pd.read_excel(RAW / DATASETS["credit_default"]["file"], header=1)
    df = df.rename(columns={"default payment next month": "target"})
    y = df.pop("target").astype(int).to_numpy()
    df = df.drop(columns=["ID"]).reset_index(drop=True)
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric = [c for c in df.columns if c not in categorical]
    meta = pd.DataFrame({
        "sex": df["SEX"].map({1: "male", 2: "female"}),
        "age_group": pd.cut(df["AGE"], [0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"]),
    })
    return RawTable("credit_default", df, y, categorical, numeric, None, meta)


def load_bank_marketing() -> RawTable:
    df = pd.read_csv(RAW / DATASETS["bank_marketing"]["file"], sep=";")
    y = (df.pop("y") == "yes").astype(int).to_numpy()
    df = df.drop(columns=["duration"])  # known only after the call ends: leakage
    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in df.columns if c not in categorical]
    return RawTable("bank_marketing", df.reset_index(drop=True), y, categorical, numeric, None)


def load_fraud() -> RawTable:
    path = RAW / DATASETS["fraud"]["file"]
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    cols = [line.split()[1].strip("'") for line in lines[:start] if line.lower().startswith("@attribute")]
    df = pd.read_csv(path, skiprows=start, header=None, names=cols, quotechar="'")
    y = df.pop("Class").astype(int).to_numpy()
    order = df["Time"].to_numpy()
    df = df.drop(columns=["Time"])  # used for ordering only, not as a feature
    return RawTable("fraud", df.reset_index(drop=True), y, [], list(df.columns), order)


LOADERS = {"credit_default": load_credit_default, "bank_marketing": load_bank_marketing, "fraud": load_fraud}

_TABLE_CACHE: dict[str, RawTable] = {}


def load_table(name: str) -> RawTable:
    if name not in _TABLE_CACHE:
        _TABLE_CACHE[name] = LOADERS[name]()
    return _TABLE_CACHE[name]


# ----------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Setting:
    dataset: str
    rate: str  # "natural" | "0.05" | "0.01"

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.rate}"

    @staticmethod
    def parse(text: str) -> "Setting":
        dataset, rate = text.split(":")
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
        if rate != "natural":
            float(rate)
        return Setting(dataset, rate)


def subsample_positives(y: np.ndarray, target_rate: float, seed: int) -> np.ndarray:
    """Row positions to keep so that positives make up `target_rate` of the kept rows."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos_keep = int(round(target_rate * len(neg) / (1 - target_rate)))
    if n_pos_keep > len(pos):
        raise ValueError(f"target rate {target_rate} is above the natural rate {len(pos) / len(y):.4f}")
    rng = np.random.default_rng(seed)
    keep = np.concatenate([neg, rng.choice(pos, n_pos_keep, replace=False)])
    return np.sort(keep)


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    cal: np.ndarray
    test: np.ndarray
    duplicates_train_test: int

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "cal": self.cal, "test": self.test}


def make_split(table: RawTable, setting: Setting, seed: int) -> Split:
    """60/10/10/20, stratified (or by time for fraud).  Positives are thinned first."""
    if setting.rate == "natural":
        rows = np.arange(len(table.y))
    else:
        rows = subsample_positives(table.y, float(setting.rate), seed)
    if table.order is not None:
        ordered = rows[np.argsort(table.order[rows], kind="stable")]
        n = len(ordered)
        train, val, cal, test = np.split(ordered, [int(0.6 * n), int(0.7 * n), int(0.8 * n)])
    else:
        train, rest = train_test_split(rows, test_size=0.4, stratify=table.y[rows], random_state=seed)
        val, rest = train_test_split(rest, test_size=0.75, stratify=table.y[rest], random_state=seed)
        cal, test = train_test_split(rest, test_size=2 / 3, stratify=table.y[rest], random_state=seed)
    return Split(train, val, cal, test, _count_duplicates(table.frame, train, test))


def _count_duplicates(frame: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> int:
    train_rows = set(map(tuple, frame.iloc[train].to_numpy().tolist()))
    return sum(tuple(r) in train_rows for r in frame.iloc[test].to_numpy().tolist())


# ----------------------------------------------------------------------------- preprocessing
class Preprocessor:
    """One-hot + standardization (+ optional top-k selection), fitted on the training split only."""

    def __init__(self, table: RawTable):
        self.table = table
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.selected: np.ndarray | None = None

    def fit(self, train_rows: np.ndarray) -> "Preprocessor":
        frame = self.table.frame.iloc[train_rows]
        cat_names: list[str] = []
        if self.table.categorical:
            self.encoder.fit(frame[self.table.categorical])
            cat_names = list(self.encoder.get_feature_names_out(self.table.categorical))
        self.scaler.fit(frame[self.table.numeric].to_numpy(dtype=np.float64))
        self.feature_names = list(self.table.numeric) + cat_names
        return self

    def transform(self, rows: np.ndarray) -> np.ndarray:
        frame = self.table.frame.iloc[rows]
        parts = [self.scaler.transform(frame[self.table.numeric].to_numpy(dtype=np.float64))]
        if self.table.categorical:
            parts.append(self.encoder.transform(frame[self.table.categorical]))
        X = np.hstack(parts).astype(np.float32)
        return X if self.selected is None else X[:, self.selected]

    def select_top_features(self, X_train: np.ndarray, y_train: np.ndarray, k: int, seed: int) -> list[str]:
        """Rank features with a small LightGBM fitted on the training split; keep the top k."""
        import lightgbm as lgb

        booster = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                     random_state=seed, verbose=-1, n_jobs=1)
        booster.fit(X_train, y_train)
        gain = booster.booster_.feature_importance(importance_type="gain")
        self.selected = np.sort(np.argsort(gain)[::-1][:k])
        self.feature_names = [self.feature_names[i] for i in self.selected]
        return self.feature_names


@dataclass
class Prepared:
    setting: Setting
    seed: int
    features: str
    feature_names: list[str]
    X: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    rows: dict[str, np.ndarray]
    duplicates_train_test: int

    @property
    def n_features(self) -> int:
        return self.X["train"].shape[1]


def prepare(setting: Setting, seed: int, features: str = "all", split_dir: Path | None = None) -> Prepared:
    table = load_table(setting.dataset)
    split = make_split(table, setting, seed)
    pre = Preprocessor(table).fit(split.train)
    X = {k: pre.transform(v) for k, v in split.as_dict().items()}
    y = {k: table.y[v] for k, v in split.as_dict().items()}
    if features == "top10":
        pre.select_top_features(X["train"], y["train"], 10, seed)
        X = {k: pre.transform(v) for k, v in split.as_dict().items()}
    elif features != "all":
        raise ValueError(f"unknown feature set {features!r}")
    if split_dir is not None:
        _save_split(split_dir, setting, seed, features, split, y, pre.feature_names)
    return Prepared(setting, seed, features, pre.feature_names, X, y, split.as_dict(), split.duplicates_train_test)


def _save_split(split_dir: Path, setting: Setting, seed: int, features: str, split: Split,
                y: dict[str, np.ndarray], feature_names: list[str]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{setting.dataset}_{setting.rate}_seed{seed}"
    npz = split_dir / f"{stem}.npz"
    if not npz.exists():
        np.savez_compressed(npz, **split.as_dict())
    info = {
        "setting": setting.key, "seed": seed, "features": features,
        "sizes": {k: int(len(v)) for k, v in split.as_dict().items()},
        "positives": {k: int(v.sum()) for k, v in y.items()},
        "duplicates_train_test": split.duplicates_train_test,
        "feature_names": feature_names,
    }
    (split_dir / f"{stem}_{features}.json").write_text(json.dumps(info, indent=2))
