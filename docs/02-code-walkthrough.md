# Code walkthrough

Read the modules in this order; each builds on the previous one.

```
src/nfi/
  data.py              raw tables -> settings -> splits -> matrices
  metrics.py           everything that turns (y, p) into numbers
  models/
    torch_train.py     shared PyTorch training loop
    tsk.py             HTSK fuzzy classifier
    mlp.py             parameter-matched MLP
    trees.py           LightGBM
    linear.py          logistic regression
    __init__.py        registry: name -> class
  search.py            random hyperparameter configurations
  experiment.py        the loop, run log, resume, ablations
  report.py            CSV tables, Markdown report, figures
  __main__.py          CLI: run / report / datasets
```

## data.py

**`RawTable`** is the only thing a loader returns: a pandas frame of features
(RangeIndex, raw row order), a 0/1 label array, the list of categorical columns,
the list of numeric columns, an optional `order` array for time splits and a
`meta` frame with subgroup columns.

**Loaders.** `load_credit_default` reads the UCI `.xls` (header on the second
row), drops `ID`, maps sex and age to subgroup labels. `load_bank_marketing`
drops `duration`. `load_fraud` parses the ARFF by hand (it is just a CSV after
the `@data` line), pops `Class`, keeps `Time` only as the ordering key.

**`Setting`** is `(dataset, rate)`; `rate` is the string `"natural"`, `"0.05"` or
`"0.01"` so it can be a dictionary key and a file name.

**`subsample_positives(y, target_rate, seed)`** keeps all negatives and
`round(rate * n_neg / (1 - rate))` positives. It raises if the target is above
the natural rate, which is how the fraud setting is protected from a typo.

**`make_split`** returns four index arrays into the raw table. The stratified
branch uses `train_test_split` three times (40% rest -> 10/30 -> 10/20). The
time branch sorts by `order` and cuts at 60/70/80%.

**`Preprocessor`** is fitted once per (setting, seed) on the training rows. It
holds a `OneHotEncoder(handle_unknown="ignore")` and a `StandardScaler`. Numeric
columns come first, then one-hot columns, which is why `feature_names` is built
the same way. `select_top_features` fits a 200-tree LightGBM on the training
matrix, takes gain importance, and stores the sorted column indices in
`self.selected`; `transform` then slices.

**`prepare(setting, seed, features, split_dir)`** glues it together and returns a
`Prepared` object with `X[part]`, `y[part]`, `rows[part]` for `train`, `val`,
`cal`, `test`. If `split_dir` is given it saves the indices (`.npz`) and a JSON
summary (sizes, positives, duplicates, feature names).

## metrics.py

- `expected_calibration_error(y, p, n_bins=15)`: sort by `p`, split into 15
  equal-size bins, sum `|mean(p) - mean(y)|` weighted by bin size.
- `calibration_slope_intercept`: two small logistic regressions fitted with
  `scipy.optimize.minimize` on `z = logit(p)`. Slope = coefficient of `z` with a
  free intercept; intercept = the constant when the slope is pinned at 1.
- `choose_threshold(y_cal, p_cal)`: `precision_recall_curve` on the calibration
  split, pick the threshold with the highest F1.
- `PlattScaler`: `LogisticRegression(C=1e6)` on the single feature `logit(p)`;
  fitted on the calibration split, applied to test.
- `evaluate(y_cal, p_cal, y_test, p_test)`: the one dictionary the runner stores
  per final fit, raw and `platt_*` versions side by side.

## models/torch_train.py

`train(model, X_train, y_train, X_val, y_val, ...)`:

1. `BCEWithLogitsLoss(pos_weight=...)` where `pos_weight = n_neg / n_pos` when
   class weighting is on, else `None`.
2. Adam with weight decay.
3. Each epoch: shuffle with a seeded generator, mini-batches, `loss + reg_weight
   * model.regularizer()`.
4. After each epoch: validation PR-AUC. Keep the best state dict; stop after
   `patience` epochs without improvement; reload the best.

`base_rate_log_odds(y, class_weight)` returns `log(p/(1-p))` of the training
rate, or 0 when weighting is on (the effective prior is then 1:1).

## models/tsk.py

```
firing:   log f_r(x) = mean_d [ -(x_d - c_rd)^2 / (2 sigma_rd^2) ]      (HTSK)
          fbar = softmax_r(log f)
rule out: y_r = w_r . x + b_r   -> BatchNorm1d over rules
logit:    sum_r fbar_r * y_r
UR:       sum_r (mean_batch(fbar_r) - 1/R)^2
```

`centers` and `log_sigma` are `nn.Parameter`s with `requires_grad=learn_mf`, so
the "fixed membership functions" ablation is one flag. `init_output_bias` writes
the base-rate log-odds into the BatchNorm bias (or the consequent bias without
BatchNorm); because the firing levels sum to one, the initial logit equals that
value for every input.

`TSKModel._init_rules` runs k-means (`n_init=1`, seeded) on the training matrix
for centres and uses `sigma_scale * std(X)` per feature for widths.

`size_info` reports rules, antecedents per rule (= number of features), membership
parameters (`2 * R * D`), consequent parameters (`R * (D + 1)`), total and
trainable parameters.

## models/mlp.py

`tsk_parameter_budget(R, D) = R*2D + R*(D+1)`. `hidden_for_budget` scans hidden
widths until the MLP parameter count passes the budget and returns the closest.
The MLP is `[Linear, ReLU, Dropout] x n_layers + Linear(hidden, 1)`; its last
bias is set to the base-rate log-odds like the TSK.

## models/trees.py and linear.py

LightGBM: `n_estimators=2000` with `early_stopping(50)` on validation
`average_precision`, `scale_pos_weight` for weighting, `n_jobs=threads`.
`size_info` dumps the model to count trees and leaves.

Logistic regression: `LogisticRegression(C, class_weight, solver="lbfgs",
max_iter=2000)`; parameters = features + 1.

## search.py

One sampler per model drawing from a seeded `numpy.random.Generator`.
`configurations(model, n, seed)` is deterministic, so the same 20 configurations
are reproduced from the run log alone. The seed is derived from the setting and
model name in `Runner.search`, so each setting gets its own draw.

## experiment.py

- `Experiment.load(yaml)` reads `configs/*.yaml` into a dataclass. Unknown keys
  go to `extra` so a config can carry notes.
- `run_key(...)` is a 16-hex-character SHA-1 of the JSON of everything that
  determines a fit.
- `RunLog` reads every `results/runs/<name>*.jsonl`, keeps finished records in a
  dict by key, appends new records to this process's own file.
- `Runner.fit_one` is the unit of work: look up the key, else prepare data (cached
  per setting/seed/feature set), build the model, time the fit, time prediction,
  evaluate, save predictions, append the record, print one line. Exceptions are
  caught and logged as `status: failed` so a long run survives a bad fit.
- `Runner.search` -> `Runner.final` -> `Runner.ablate` is the protocol.
  `ablate` writes `<name>.<setting>.ablations.json` mapping variant name to the
  list of run keys, which is how the report finds them without re-deriving
  configurations.

## report.py

`load_frame` flattens the run log into one row per final/ablation fit.
`mean_std_table` produces the `mean ± std` strings. `paired_differences` reloads
the saved predictions of TSK and the baseline for the same seed (same test rows),
computes the metric difference, and bootstraps over test rows 500 times; the
bootstrap matrices are averaged over seeds before taking the 2.5 and 97.5
percentiles. `weighting_effect` subtracts unweighted from weighted per
(setting, model, seed). `subgroup_table` uses the saved `rows_test` to look up
sex and age in the raw table. `verdict_table` applies the section 6.4 rules.
Figures use a fixed colour per model (never cycled), thin lines, direct labels
and a legend.

## Extending it

- New model: add a class with `fit / predict_proba / size_info / fit_info`, a
  sampler in `search.py`, and register both in `models/__init__.py` and
  `search.SAMPLERS`.
- New dataset: a loader returning `RawTable`, an entry in `DATASETS` and
  `LOADERS`, and a setting string in the YAML.
- New metric: add it to `evaluate` in `metrics.py`; it appears in the run log and
  `all_final_runs.csv` automatically; add it to `MAIN_METRICS` for the table.
