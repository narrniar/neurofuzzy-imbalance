# Neuro-fuzzy classifiers under class imbalance

Does a gradient-trained TSK fuzzy classifier (HTSK, Cui et al. 2021) give a better
balance of minority-class ranking (PR-AUC), calibration and model size than
LightGBM, logistic regression and a parameter-matched MLP as the positive class
becomes rarer?  This repository runs that experiment end to end on a laptop CPU.

## Setup (macOS)

```bash
bash scripts/setup_macos.sh            # .venv on python3.13, deps, LightGBM/PyTorch OpenMP fix
.venv/bin/python scripts/download_data.py   # 3 public datasets -> data/raw, checksums -> data/checksums.json
.venv/bin/python -m nfi datasets       # sanity: sizes, positive rates, split sizes
.venv/bin/python -m pytest -q tests    # 20 behavioural tests on synthetic data
```

Requirements: Homebrew `python@3.13` (`brew install python@3.13`), Xcode command line
tools (`install_name_tool`, `codesign`).  No GPU.  The fraud ARFF is 150 MB.

Why the OpenMP fix: the LightGBM and PyTorch wheels each ship `libomp.dylib`; two
OpenMP runtimes in one process make LightGBM segfault (exit 139) or PyTorch deadlock
after a LightGBM fit.  `setup_macos.sh` repoints LightGBM's rpath at PyTorch's copy.

## Running

```bash
# full protocol: 7 settings x 4 models x 2 weightings x 20 configs, 5 seeds, ablations
bash scripts/run_parallel.sh configs/full.yaml 2        # one process per setting, 2 threads each
# or a single process, resumable, any subset:
.venv/bin/python -m nfi run configs/full.yaml --settings fraud:natural --models tsk,lgbm
# tables + figures -> results/report/full/REPORT.md
.venv/bin/python -m nfi report configs/full.yaml
```

`configs/quick.yaml` (5 configs, 3 seeds) and `configs/smoke.yaml` (pipeline check,
about a minute) use the same code path.  Every fit is logged as one JSON line in
`results/runs/` with run id, timestamp, setting, seed, configuration, status,
metrics, timings and the path of its saved predictions; re-running skips finished
fits, so an interrupted run resumes.

## What the code does

| Module | Content |
|---|---|
| `nfi/data.py` | loaders, positive-class thinning to 5% / 1%, stratified 60/10/10/20 split (time split for fraud), one-hot + standardisation fitted on train only, top-10 feature ranking on train only, duplicate check train vs test, `duration` dropped from Bank Marketing |
| `nfi/models/tsk.py` | HTSK: Gaussian MFs, k-means centres, mean-over-D log firing, first-order consequents, BatchNorm on consequents, uniform regularisation, learned or fixed MFs |
| `nfi/models/mlp.py` | MLP with hidden width chosen to match the TSK parameter budget |
| `nfi/models/trees.py`, `linear.py` | LightGBM (early stopping on validation PR-AUC), L2 logistic regression |
| `nfi/search.py` | random configurations per model; selection by validation PR-AUC on seed 0 |
| `nfi/metrics.py` | PR-AUC, ROC-AUC, Brier, ECE (15 equal-size bins), calibration slope and intercept, Platt scaling on the calibration split, F1 threshold from the calibration split |
| `nfi/experiment.py` | search -> final seeds -> ablations (fixed MFs, 4/8/16 rules, UR on/off, top-10 features for every model) |
| `nfi/report.py` | mean ± std tables, paired bootstrap CIs vs LightGBM and LogReg, SQ1 figure, reliability plots, weighting effect, subgroup errors (sex, age), confident false negatives, verdict per section 6.4 |

Class weighting = `pos_weight` / `scale_pos_weight` / `class_weight="balanced"`
with the negative:positive ratio of the training split.  Both PyTorch models start
with their output bias at the base-rate log-odds, so no model is penalised on
calibration merely for starting at 0.5.

## Reproducibility

Seeds fix NumPy, PyTorch, LightGBM and the splits; split row indices are saved in
`results/splits/`, dataset checksums in `data/checksums.json`, package versions via
`.venv/bin/pip freeze`.  Every number in the report traces to a `key` in the run log
and a prediction file in `results/preds/`.
