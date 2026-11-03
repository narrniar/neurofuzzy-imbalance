# How to run it

## 0. Requirements

- macOS (Apple Silicon or Intel) with Homebrew; Linux works too but skip the
  OpenMP patch (the Linux wheels share one runtime).
- `brew install python@3.13 libomp`
- Xcode command line tools (`xcode-select --install`) for `install_name_tool`
  and `codesign`.

## 1. Environment

```bash
git clone git@github.com:narrniar/neurofuzzy-imbalance.git
cd neurofuzzy-imbalance
bash scripts/setup_macos.sh
```

The script creates `.venv`, installs the package in editable mode (`pip install
-e .`), repoints LightGBM's rpath at PyTorch's `libomp.dylib` and prints the
installed versions. Re-run it after any `pip install --upgrade lightgbm`.

## 2. Data

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python -m nfi datasets
```

The second command prints, for every dataset and rate, the rows and positives of
each split and the number of duplicate rows between train and test. Compare the
checksums in `data/checksums.json` with yours if you want to be sure you have the
same bytes.

## 3. Tests

```bash
.venv/bin/python -m pytest -q tests
```

Twenty tests on synthetic data, about three seconds.

## 4. Experiments

Three configuration files, same code path:

| File | Configs | Seeds | Epochs / patience | Purpose | Time on M4 Pro |
|---|---|---|---|---|---|
| `configs/smoke.yaml` | 1 | 1 | 3 / 2 | does the pipeline work | 1 min |
| `configs/quick.yaml` | 5 | 3 | 40 / 6 | first look | 5 min |
| `configs/full.yaml` | 20 | 5 | 100 / 10 | the protocol | 15 min |

Parallel (one process per setting, recommended):

```bash
bash scripts/run_parallel.sh configs/full.yaml 2    # second argument = threads per process
```

Single process, any subset, resumable:

```bash
.venv/bin/python -m nfi run configs/full.yaml
.venv/bin/python -m nfi run configs/full.yaml --settings fraud:natural --models tsk,lgbm --threads 8
```

Progress is one line per fit:

```
[search  ] credit_default:0.05    tsk    cw=0 feat=all   seed=0 val_pr_auc=0.2398 (1.6s)
[final   ] credit_default:0.05    tsk    cw=0 feat=all   seed=3 test_pr_auc=0.2311 (2.1s)
[ablation] credit_default:0.05    tsk    cw=0 feat=top10 seed=0 test_pr_auc=0.1960 (1.2s)
```

Interrupt with Ctrl-C at any time; re-running the same command skips every fit
that already has a record.

## 5. Report

```bash
.venv/bin/python -m nfi report configs/full.yaml
open results/report/full/REPORT.md
```

Outputs in `results/report/<name>/`:

| File | Content |
|---|---|
| `REPORT.md` | everything below, as one Markdown page |
| `main_results.csv` | mean ± std per setting, model, weighting |
| `all_final_runs.csv` | one row per final/ablation fit, every metric |
| `model_size_and_resources.csv` | parameters, rules, trees, fit time, memory |
| `paired_differences.csv` | TSK minus LightGBM / LogReg with bootstrap CIs |
| `weighting_effect.csv` | weighted minus unweighted per metric (SQ3) |
| `ablations.csv` | TSK variants and the top-10-feature runs (SQ2) |
| `subgroups_credit_default.csv` | PR-AUC, recall, FNR by sex and age group |
| `confident_false_negatives_credit_default.csv` | the 10 defaulters each model was most sure about |
| `verdict.csv` | section 6.4 rule applied per setting |
| `fig_imbalance.png` | PR-AUC and ECE vs positive rate (SQ1) |
| `fig_reliability.png` | reliability curves per setting |

## 6. Where everything is stored

```
results/
  runs/     full.<setting>.jsonl        one JSON line per fit (the source of truth)
            full.<setting>.ablations.json
  preds/    <key>.npz                   p_val, p_cal, p_test, y_cal, y_test, rows_test
  splits/   <dataset>_<rate>_seed<k>.npz  row indices of the four splits
  report/   <name>/                     tables and figures
  logs/     <name>.<setting>.log        stdout of each parallel process
```

`runs/`, `preds/`, `splits/`, `logs/` are git-ignored (106 MB of predictions);
`report/` is committed.

## 7. Tracing a number back

Every row in `all_final_runs.csv` has a `key`. `grep <key> results/runs/*.jsonl`
gives the full record: timestamp, configuration, seed, epochs, timings.
`results/preds/<key>.npz` holds its predictions, and `rows_test` inside it indexes
the raw table, so `load_table(name).frame.iloc[rows_test]` shows the original
rows.

## 8. Changing the protocol

Edit a YAML or copy one. Fields: `name`, `seeds`, `n_configs`, `max_epochs`,
`patience`, `threads`, `settings`, `models`, `weighting`, `ablations`. A new
`name` writes to new log files, so old results are never overwritten.
