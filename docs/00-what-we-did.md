# What we did, step by step (2026-11-04)

This is the narrative log of how the project was built in one session, including
the dead ends. Read it to understand *why* the code looks the way it does.

## 1. Read the proposal and decided the architecture

The proposal (section 4 to 6) defines a fixed protocol: 7 dataset settings, 4
models, class weighting on/off, 20 random configurations each, best configuration
retrained on 5 seeds, TSK ablations, calibration and ranking metrics. We turned that
into one Python package (`src/nfi`) with one module per concern:

| Concern | Module |
|---|---|
| load data, thin positives, split, preprocess | `data.py` |
| the four models behind one interface | `models/` |
| hyperparameter spaces | `search.py` |
| metrics and Platt scaling | `metrics.py` |
| the experiment loop and run log | `experiment.py` |
| tables and figures | `report.py` |
| command line | `__main__.py` |

Rule we followed: every model has the same four methods (`fit`, `predict_proba`,
`size_info`, `fit_info`) so the experiment loop never knows which model it holds.

## 2. Set up the environment

- Machine: Apple M4 Pro, 12 cores, 24 GB RAM, macOS 15, no GPU.
- Python: Homebrew `python3.13` in a virtualenv (`.venv`). The proposal says 3.11;
  3.13 was what the machine had with good wheel coverage.
- Packages: numpy 2.5, pandas 3.1, scikit-learn 1.9, lightgbm 4.7, torch 2.15,
  matplotlib, pyyaml, xlrd (for the UCI `.xls`), scipy, tabulate, pytest.

## 3. Downloaded the data with checksums

`scripts/download_data.py` fetches the two UCI zips and the fraud table. The Kaggle
page for the fraud data needs a login, so we used the OpenML mirror (dataset 1597,
same 284,807 x 31 table). SHA-256 of every file is written to `data/checksums.json`.

## 4. Wrote the data layer

- `load_*` functions return a `RawTable`: features, labels, categorical column
  names, an optional time key (fraud) and subgroup columns (credit default: sex, age).
- `subsample_positives` removes positives at random until the target rate is hit.
  Negatives are never touched, so 5% and 1% versions differ only in how many
  positives remain.
- `make_split` does 60/10/10/20 stratified (fraud: sorted by time and cut), and
  counts exact duplicate rows between train and test.
- `Preprocessor` fits one-hot and standardisation on the training rows only.
  `select_top_features` ranks features with a small LightGBM fitted on train only.
- Split row indices are saved to `results/splits/` so every prediction can be
  traced back to a raw row.

## 5. Wrote the models

- `tsk.py`: HTSK. Rule centres from k-means on the training split, widths from the
  per-feature standard deviation times a tunable scale. Log-firing is the *mean* of
  the per-dimension Gaussian exponents (the HTSK trick, Cui et al. 2021), normalised
  with a softmax. Consequents are a `Linear(D, R)` layer, BatchNorm over rules,
  mixed with the normalised firing. Uniform regularisation is the squared gap
  between the mean firing of each rule and 1/R.
- `mlp.py`: hidden width chosen so the parameter count matches a TSK with the same
  rule count (`hidden_for_budget`).
- `trees.py`: LightGBM with early stopping on validation PR-AUC.
- `linear.py`: L2 logistic regression.
- `torch_train.py`: one training loop for TSK and MLP: Adam, BCE-with-logits with
  optional `pos_weight`, early stopping on validation PR-AUC, restore best epoch.

## 6. Wrote the experiment runner

- A run is keyed by a hash of (setting, model, weighting, feature set, config,
  seed, evaluate-test flag). Finished keys are skipped, so an interrupted run
  resumes and identical ablation variants are not refitted.
- Each fit appends one JSON line to `results/runs/<name>.<setting>.jsonl` with
  run id, timestamp, config, seed, status, metrics, timings, size and the path of
  the saved predictions (`results/preds/<key>.npz`).
- Phases: `search` (20 configs on seed 0, validation only) -> `final` (best config,
  5 seeds, full evaluation on test) -> `ablation` (TSK variants + top-10 features
  for every model).

## 7. Wrote the metrics

PR-AUC, ROC-AUC, Brier, log loss, ECE with 15 equal-size bins, calibration slope
(logistic regression of y on logit p) and intercept (slope fixed at 1), threshold
chosen by F1 on the calibration split, precision / recall / F1 on test, Platt
scaling fitted on the calibration split and applied to test.

## 8. Smoke-tested and hit three macOS problems

1. `import lightgbm` failed: the wheel needs Homebrew `libomp`.
2. After `brew install libomp`, the first LightGBM fit segfaulted (exit 139)
   whenever torch was imported first. Importing LightGBM first stopped the crash...
3. ...but then the first PyTorch fit after a LightGBM fit deadlocked at 0% CPU.
   Both wheels ship their own `libomp.dylib`; two OpenMP runtimes in one process
   is undefined behaviour. Fix: `install_name_tool -rpath` to make LightGBM load
   PyTorch's copy, then re-sign the dylib. This is in `scripts/setup_macos.sh`.

Also: pandas 3 stores strings as a `str` dtype, so `dtype == object` no longer
finds categorical columns; LightGBM 4.7 renamed `eval_set` to `eval_X` / `eval_y`.

## 9. Found a fairness problem in the TSK and fixed it

With BatchNorm on the rule outputs, the TSK starts at probability 0.5 for every
row. Early stopping on PR-AUC (a ranking metric) does not care about the
intercept, so at 1% positives the TSK could stop with a badly biased intercept
and lose on calibration for a reason unrelated to fuzzy rules. We initialise the
output bias of both PyTorch models to the base-rate log-odds (0 when class
weighting is on). This is standard "prior probability" initialisation.

## 10. Ran the full protocol

`bash scripts/run_parallel.sh configs/full.yaml 2`: seven processes, one per
setting, two threads each. 1,680 fits, zero failures, about 15 minutes. The
proposal estimated under 40 CPU hours; the models are small, so fits take 0.2 to
8 seconds.

## 11. Built the report and read it critically

See `docs/04-results-and-interpretation.md`. Two protocol weaknesses surfaced
(early stopping on a 24-positive validation split; configuration selection on a
single seed). They are documented as open items, not patched silently.

## 12. Tests

`tests/test_pipeline.py`: 20 behavioural tests on synthetic data (subsampling
rate, disjoint and stratified splits, time split never trains on the future,
preprocessing fitted on train only, ECE and slope recover known distortions,
Platt improves an over-confident model, every model beats chance with and
without weighting, fixed MFs do not move, MLP budget matching).
