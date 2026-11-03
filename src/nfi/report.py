"""Aggregate the run log into the tables and figures of the evaluation plan.

Outputs go to results/report/<experiment>/: REPORT.md plus CSV tables and PNG figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import ROOT, load_table  # noqa: E402
from .experiment import Experiment, RunLog  # noqa: E402
from .metrics import expected_calibration_error, reliability_curve  # noqa: E402

MODEL_ORDER = ["tsk", "lgbm", "logreg", "mlp"]
MODEL_LABEL = {"tsk": "TSK (HTSK)", "lgbm": "LightGBM", "logreg": "LogReg", "mlp": "MLP"}
# Fixed categorical hue per model (validated reference palette, slots 1-4), never cycled.
MODEL_COLOR = {"tsk": "#2a78d6", "lgbm": "#eb6834", "logreg": "#1baf7a", "mlp": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MAIN_METRICS = ["pr_auc", "roc_auc", "brier", "ece", "cal_slope", "cal_intercept", "platt_brier", "platt_ece", "f1"]
N_BOOTSTRAP = 500


# ----------------------------------------------------------------------------- loading
def load_frame(exp: Experiment) -> pd.DataFrame:
    log = RunLog(exp.runs_file(), exp.all_runs_files())
    recs = [r for r in log.records.values() if "test" in r]
    rows = []
    for r in recs:
        row = {k: r[k] for k in ["key", "phase", "setting", "dataset", "rate", "model", "class_weight", "features",
                                  "seed", "n_features", "fit_seconds", "predict_ms_per_1k", "peak_rss_mb",
                                  "preds_path", "duplicates_train_test"]}
        row.update(r["test"])
        row["n_parameters"] = r["size"].get("n_parameters")
        row["n_rules"] = r["size"].get("n_rules")
        row["n_trees"] = r["size"].get("n_trees")
        row["n_leaves"] = r["size"].get("n_leaves")
        row["positives_test"] = r["positives"]["test"]
        row["test_size"] = r["split_sizes"]["test"]
        row["config"] = json.dumps(r["config"], sort_keys=True)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no finished test runs in {exp.runs_dir}")
    df["setting"] = pd.Categorical(df["setting"], categories=exp.settings, ordered=True)
    return df


def final_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["phase"] == "final") & (df["features"] == "all")]


def fmt(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def mean_std_table(df: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    agg = df.groupby(group, observed=True)[metrics].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        out[m] = [fmt(a, b if not np.isnan(b) else 0.0) for a, b in zip(agg[(m, "mean")], agg[(m, "std")])]
    out["n_seeds"] = df.groupby(group, observed=True).size()
    return out.reset_index()


# ----------------------------------------------------------------------------- paired differences
def _ap(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="stable")
    y = y[order]
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / n_pos)


def _metric_triplet(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([_ap(y, p), float(np.mean((p - y) ** 2)), expected_calibration_error(y, p)])


def paired_differences(df: pd.DataFrame, baseline: str, n_boot: int = N_BOOTSTRAP) -> pd.DataFrame:
    """TSK minus baseline on PR-AUC, Brier and ECE, same test rows, bootstrap over test instances
    (paired), pooled over seeds.  Negative Brier/ECE differences favour TSK."""
    rows = []
    for (setting, cw), part in df.groupby(["setting", "class_weight"], observed=True):
        tsk = part[part["model"] == "tsk"].set_index("seed")
        base = part[part["model"] == baseline].set_index("seed")
        seeds = sorted(set(tsk.index) & set(base.index))
        if not seeds:
            continue
        rng = np.random.default_rng(0)
        point, boots = [], []
        for seed in seeds:
            a, b = np.load(ROOT / tsk.loc[seed, "preds_path"]), np.load(ROOT / base.loc[seed, "preds_path"])
            y, pa, pb = a["y_test"], a["p_test"], b["p_test"]
            point.append(_metric_triplet(y, pa) - _metric_triplet(y, pb))
            idx = rng.integers(0, len(y), size=(n_boot, len(y)))
            boots.append(np.array([_metric_triplet(y[i], pa[i]) - _metric_triplet(y[i], pb[i]) for i in idx]))
        point = np.mean(point, axis=0)
        pooled = np.mean(boots, axis=0)
        lo, hi = np.nanpercentile(pooled, 2.5, axis=0), np.nanpercentile(pooled, 97.5, axis=0)
        for j, m in enumerate(["pr_auc", "brier", "ece"]):
            rows.append({"setting": setting, "class_weight": cw, "baseline": baseline, "metric": m,
                         "tsk_minus_baseline": point[j], "ci_low": lo[j], "ci_high": hi[j], "n_seeds": len(seeds)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=9)


def figure_imbalance(df: pd.DataFrame, out: Path) -> None:
    """SQ1: PR-AUC and ECE against the positive rate, one panel per dataset (unweighted models)."""
    datasets = [d for d in ["credit_default", "bank_marketing"] if d in set(df["dataset"])]
    if not datasets:
        return
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.2 * len(datasets), 7.2), squeeze=False, facecolor=SURFACE)
    for col, dataset in enumerate(datasets):
        part = df[(df["dataset"] == dataset) & (~df["class_weight"])]
        for row, metric in enumerate(["pr_auc", "ece"]):
            ax = axes[row, col]
            _style(ax)
            for model in MODEL_ORDER:
                sub = part[part["model"] == model].groupby("rate", observed=True)[metric].agg(["mean", "std"])
                if sub.empty:
                    continue
                order = ["natural", "0.05", "0.01"]
                sub = sub.reindex([r for r in order if r in sub.index])
                x = np.arange(len(sub))
                ax.errorbar(x, sub["mean"], yerr=sub["std"].fillna(0), color=MODEL_COLOR[model], linewidth=2,
                            marker="o", markersize=6, capsize=3, label=MODEL_LABEL[model])
                ax.annotate(MODEL_LABEL[model], (x[-1], sub["mean"].iloc[-1]), xytext=(6, 0),
                            textcoords="offset points", fontsize=8, color=INK2, va="center")
                ax.set_xticks(x, [f"{r}" for r in sub.index])
            ax.set_title(f"{dataset} - {metric.replace('_', '-').upper()}", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("positive rate", color=INK2, fontsize=9)
            ax.set_xlim(-0.3, len(x) - 1 + 1.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Ranking and calibration as the positive class becomes rarer (no class weighting)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_reliability(df: pd.DataFrame, out: Path) -> None:
    """Reliability plots, seed 0, no class weighting, raw probabilities, one panel per setting."""
    settings = [s for s in df["setting"].cat.categories if s in set(df["setting"])]
    n = len(settings)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows), squeeze=False, facecolor=SURFACE)
    for k, setting in enumerate(settings):
        ax = axes[k // cols, k % cols]
        _style(ax)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
        part = df[(df["setting"] == setting) & (~df["class_weight"]) & (df["seed"] == df["seed"].min())]
        top = 0.05
        for model in MODEL_ORDER:
            sub = part[part["model"] == model]
            if sub.empty:
                continue
            pred = np.load(ROOT / sub.iloc[0]["preds_path"])
            curve = reliability_curve(pred["y_test"], pred["p_test"])
            ax.plot(curve["mean_pred"], curve["frac_pos"], color=MODEL_COLOR[model], linewidth=2, marker="o",
                    markersize=5, label=MODEL_LABEL[model])
            top = max(top, max(curve["mean_pred"] + curve["frac_pos"]))
        ax.set_xlim(0, min(1, top * 1.1))
        ax.set_ylim(0, min(1, top * 1.1))
        ax.set_title(setting, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("mean predicted probability", fontsize=8, color=INK2)
        ax.set_ylabel("fraction of positives", fontsize=8, color=INK2)
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Reliability (15 equal-size bins, seed 0, no weighting, before Platt scaling)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- SQ2 / SQ3 / subgroups
def ablation_table(exp: Experiment, df: pd.DataFrame) -> pd.DataFrame:
    index: dict = {}
    for path in exp.all_ablation_files():
        index.update(json.loads(path.read_text()))
    if not index:
        return pd.DataFrame()
    by_key = df.set_index("key")
    rows = []
    for setting, variants in index.items():
        for variant, keys in variants.items():
            part = by_key.loc[[k for k in keys if k in by_key.index]]
            if part.empty:
                continue
            model, name = variant.split("|")
            rows.append({"setting": setting, "model": model, "variant": name,
                         "pr_auc": fmt(part["pr_auc"].mean(), part["pr_auc"].std(ddof=0)),
                         "brier": fmt(part["brier"].mean(), part["brier"].std(ddof=0), 4),
                         "ece": fmt(part["ece"].mean(), part["ece"].std(ddof=0), 4),
                         "n_parameters": int(part["n_parameters"].mean()), "n_seeds": len(part)})
    return pd.DataFrame(rows)


def weighting_effect(df: pd.DataFrame) -> pd.DataFrame:
    """SQ3: class-weighted minus unweighted, averaged over seeds, per setting and model."""
    metrics = ["pr_auc", "brier", "ece", "cal_intercept", "cal_slope", "platt_ece", "recall", "f1"]
    on = df[df["class_weight"]].set_index(["setting", "model", "seed"])[metrics]
    off = df[~df["class_weight"]].set_index(["setting", "model", "seed"])[metrics]
    delta = (on - off).dropna().reset_index()
    out = delta.groupby(["setting", "model"], observed=True)[metrics].mean().reset_index()
    return out.rename(columns={m: f"d_{m}" for m in metrics})


def subgroup_table(df: pd.DataFrame) -> pd.DataFrame:
    part = df[(df["dataset"] == "credit_default") & (df["rate"] == "natural") & (~df["class_weight"])]
    if part.empty:
        return pd.DataFrame()
    meta = load_table("credit_default").meta
    rows = []
    for model, sub in part.groupby("model"):
        for _, run in sub.iterrows():
            pred = np.load(ROOT / run["preds_path"])
            y, p, test_rows = pred["y_test"], pred["p_test"], pred["rows_test"]
            pred_pos = p >= run["threshold"]
            for column in ["sex", "age_group"]:
                groups = meta[column].iloc[test_rows].astype(str).to_numpy()
                for g in np.unique(groups):
                    m = groups == g
                    pos = y[m] == 1
                    rows.append({"model": model, "seed": run["seed"], "group_by": column, "group": g, "n": int(m.sum()),
                                 "positives": int(pos.sum()), "pr_auc": _ap(y[m], p[m]),
                                 "recall": float(pred_pos[m][pos].mean()) if pos.any() else np.nan,
                                 "false_negative_rate": float(1 - pred_pos[m][pos].mean()) if pos.any() else np.nan,
                                 "flag_rate": float(pred_pos[m].mean())})
    out = pd.DataFrame(rows).groupby(["model", "group_by", "group"]).agg(
        n=("n", "mean"), positives=("positives", "mean"), pr_auc=("pr_auc", "mean"),
        recall=("recall", "mean"), false_negative_rate=("false_negative_rate", "mean"),
        flag_rate=("flag_rate", "mean")).reset_index()
    return out


def confident_false_negatives(df: pd.DataFrame, out_dir: Path, k: int = 10) -> pd.DataFrame:
    part = df[(df["dataset"] == "credit_default") & (df["rate"] == "natural") & (~df["class_weight"])]
    if part.empty:
        return pd.DataFrame()
    frame = load_table("credit_default").frame
    pieces = []
    for model, sub in part.groupby("model"):
        run = sub.sort_values("seed").iloc[0]
        pred = np.load(ROOT / run["preds_path"])
        positives = np.flatnonzero(pred["y_test"] == 1)
        worst = positives[np.argsort(pred["p_test"][positives])[:k]]
        piece = frame.iloc[pred["rows_test"][worst]].copy()
        piece.insert(0, "p_default", pred["p_test"][worst])
        piece.insert(0, "model", model)
        pieces.append(piece)
    table = pd.concat(pieces)
    table.to_csv(out_dir / "confident_false_negatives_credit_default.csv", index=False)
    return table


# ----------------------------------------------------------------------------- verdict (section 6.4)
def verdict_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, cw), part in df.groupby(["setting", "class_weight"], observed=True):
        means = part.groupby("model")[["pr_auc", "ece", "brier", "platt_ece", "n_parameters"]].mean()
        if "tsk" not in means.index:
            continue
        baselines = means.drop(index="tsk")
        best_baseline = baselines["pr_auc"].idxmax()
        gap = means.loc["tsk", "pr_auc"] - baselines["pr_auc"].max()
        better_cal = means.loc["tsk", "ece"] < baselines["ece"].min() or means.loc["tsk", "brier"] < baselines["brier"].min()
        better_cal_platt = means.loc["tsk", "platt_ece"] < baselines["platt_ece"].min()
        smaller = means.loc["tsk", "n_parameters"] < baselines.loc[best_baseline, "n_parameters"]
        worse_than_all = (means.loc["tsk", "pr_auc"] < means.loc[["lgbm", "logreg"], "pr_auc"].min()
                          and means.loc["tsk", "brier"] > means.loc[["lgbm", "logreg"], "brier"].max())
        if gap >= -0.02 and (better_cal or smaller):
            label = "supports"
        elif worse_than_all:
            label = "contradicts"
        else:
            label = "qualifies"
        rows.append({"setting": setting, "class_weight": cw, "best_baseline": best_baseline,
                     "tsk_pr_auc_gap": round(gap, 4), "within_0.02": gap >= -0.02,
                     "tsk_better_calibrated_raw": bool(better_cal), "tsk_better_calibrated_after_platt": bool(better_cal_platt),
                     "tsk_smaller_than_best_baseline": bool(smaller), "verdict": label})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- assembly
def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_no data_\n"
    return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"


def build_report(exp: Experiment) -> Path:
    out_dir = exp.out / "report" / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_frame(exp)
    final = final_runs(df)

    main = mean_std_table(final, ["setting", "model", "class_weight"], MAIN_METRICS)
    main["model"] = pd.Categorical(main["model"], MODEL_ORDER, ordered=True)
    main = main.sort_values(["setting", "class_weight", "model"])
    main.to_csv(out_dir / "main_results.csv", index=False)
    final.drop(columns=["config"]).to_csv(out_dir / "all_final_runs.csv", index=False)

    size = final.groupby(["setting", "model"], observed=True).agg(
        n_features=("n_features", "first"), n_parameters=("n_parameters", "mean"), n_rules=("n_rules", "first"),
        n_trees=("n_trees", "mean"), n_leaves=("n_leaves", "mean"), fit_seconds=("fit_seconds", "mean"),
        predict_ms_per_1k=("predict_ms_per_1k", "mean"), peak_rss_mb=("peak_rss_mb", "max")).reset_index()
    size.to_csv(out_dir / "model_size_and_resources.csv", index=False)

    diffs = pd.concat([paired_differences(final, "lgbm"), paired_differences(final, "logreg")], ignore_index=True)
    diffs.to_csv(out_dir / "paired_differences.csv", index=False)
    sq3 = weighting_effect(final)
    sq3.to_csv(out_dir / "weighting_effect.csv", index=False)
    abl = ablation_table(exp, df)
    abl.to_csv(out_dir / "ablations.csv", index=False)
    groups = subgroup_table(final)
    groups.to_csv(out_dir / "subgroups_credit_default.csv", index=False)
    confident_false_negatives(final, out_dir)
    verdict = verdict_table(final)
    verdict.to_csv(out_dir / "verdict.csv", index=False)
    figure_imbalance(final, out_dir / "fig_imbalance.png")
    figure_reliability(final, out_dir / "fig_reliability.png")

    splits = final.groupby("setting", observed=True).agg(test_size=("test_size", "first"),
                                                          positives_test=("positives_test", "first"),
                                                          duplicates_train_test=("duplicates_train_test", "first")).reset_index()
    best_cfg = final[final["seed"] == final["seed"].min()][["setting", "model", "class_weight", "config"]]

    text = [f"# Results: {exp.name}\n",
            f"Seeds: {exp.seeds}; configurations per model and setting: {exp.n_configs}; "
            f"max epochs {exp.max_epochs}, patience {exp.patience}. Values are mean ± std over seeds.\n",
            "## Data (seed 0 split)\n", _md(splits, ".0f"),
            "## Main results (test split)\n",
            "PR-AUC is the primary metric. `ece` uses 15 equal-size bins; slope 1 / intercept 0 is perfect "
            "calibration; `platt_*` are after Platt scaling fitted on the calibration split; F1 at the threshold "
            "chosen on the calibration split.\n", _md(main),
            "## Model size and resources\n",
            "`n_parameters` for LightGBM is its total leaf count (not comparable to rule parameters); "
            "`peak_rss_mb` is the process peak and only indicative.\n", _md(size, ".1f"),
            "## Paired differences, TSK minus baseline (95% bootstrap CI over test rows, pooled over seeds)\n",
            "Positive PR-AUC favours TSK; negative Brier / ECE favours TSK.\n", _md(diffs),
            "## SQ1: effect of the positive rate\n", "![imbalance](fig_imbalance.png)\n",
            "## SQ2: TSK ablations (no class weighting; `base` is the selected configuration)\n", _md(abl),
            "## SQ3: class weighting effect (weighted minus unweighted)\n", _md(sq3),
            "## Reliability\n", "![reliability](fig_reliability.png)\n",
            "## Subgroups on credit default (natural rate, unweighted, mean over seeds)\n", _md(groups),
            "Confident false negatives: `confident_false_negatives_credit_default.csv`.\n",
            "## Verdict per setting (section 6.4 rules)\n", _md(verdict),
            "## Selected configurations\n", _md(best_cfg, ".4f")]
    (out_dir / "REPORT.md").write_text("\n".join(text))
    print(f"report: {out_dir / 'REPORT.md'}")
    return out_dir
