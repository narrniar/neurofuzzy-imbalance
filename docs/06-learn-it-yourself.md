# Learn it yourself: a study path through this repository

The fastest way to own this project is to rebuild pieces of it and check your
version against the committed one. Each step names the file to read, what to
try, and how to verify.

## Week 1: data and metrics

1. **Read** `src/nfi/data.py`. Run `.venv/bin/python -m nfi datasets` and
   reproduce the numbers by hand for one dataset: how many positives survive
   thinning to 5%? (`round(0.05 * n_neg / 0.95)`.)
2. **Exercise.** Write a function that checks, for one seed, that no row index
   appears in two splits and that the positive rate of each split is within 0.5
   percentage points of the others. Compare with
   `tests/test_pipeline.py::test_split_is_disjoint_and_stratified`.
3. **Read** `src/nfi/metrics.py`. Generate perfectly calibrated probabilities
   (`y ~ Bernoulli(p)`), then double the logits, and watch ECE, slope and Brier
   move. Then apply `PlattScaler` and watch them come back.
4. **Question to answer in writing:** why is the threshold chosen on the
   calibration split and not on the test split? What would go wrong?

## Week 2: the models

5. **Read** `src/nfi/models/tsk.py` next to Cui et al. (2021). Write the HTSK
   firing formula on paper, then find each term in `HTSK.firing`. Change
   `mean(dim=-1)` to `sum(dim=-1)` (plain TSK) and run the fraud setting with
   `--models tsk`; the 29-feature model should get worse. Change it back.
6. **Exercise.** Train one TSK on credit default (see `/tmp`-style script in
   `docs/00-what-we-did.md` step 9 or write your own with `prepare` + `build`)
   and print `clf.net.centers` after training. Do the rule centres still look
   like the k-means centres? Which features moved most?
7. **Read** `models/mlp.py`. Verify `hidden_for_budget` by hand for D=33, R=8.
8. **Exercise.** Add a fifth model, for example scikit-learn's
   `HistGradientBoostingClassifier`, following "Extending it" in
   `docs/02-code-walkthrough.md`. Run `configs/smoke.yaml` with it.

## Week 3: the protocol

9. **Read** `src/nfi/experiment.py`. Follow one fit from `Runner.run` down to
   the JSON line. Open a line from `results/runs/` and name every field.
10. **Exercise (real improvement #1).** Change LightGBM early stopping to use
    log loss (`eval_metric="binary_logloss"`) and rerun only
    `--settings fraud:natural --models lgbm` under a new config `name`. Compare
    `main_results.csv` before and after. Expected: fraud PR-AUC rises from about
    0.68 to about 0.80.
11. **Exercise (real improvement #2).** Make `Runner.search` score each
    configuration on the mean validation PR-AUC of two seeds instead of one.
    Rerun the fraud TSK. Does it still pick the 16-rule configuration?
12. **Read** `src/nfi/report.py::paired_differences`. Explain why bootstrapping
    over test rows (paired) is the right unit, and why averaging the bootstrap
    matrices over seeds before taking percentiles is a simplification.

## Week 4: write-up

13. Regenerate the report after exercises 10 and 11, update
    `docs/04-results-and-interpretation.md`, and write the paper's results
    section from it. Use the section 6.4 rules, then say what they miss.
14. Produce the figures the paper needs from `all_final_runs.csv` with your own
    plotting code; the repository's figures are a starting point, not the final
    ones.

## Reading list (from the proposal, in the order that matches the code)

- Takagi & Sugeno (1985); Jang (1993): the TSK rule form and ANFIS.
- Cui, Wu & Huang (2020): mini-batch training, UR, BatchNorm for TSK.
- Cui, Wu, Xu (2021): HTSK, why Gaussian TSK fails in high dimensions.
- Ke et al. (2017): LightGBM.
- Grinsztajn et al. (2022); Shwartz-Ziv & Armon (2022): trees vs deep learning
  on tabular data.
- Saito & Rehmsmeier (2015): why PR curves for imbalance.
- Niculescu-Mizil & Caruana (2005); Guo et al. (2017): calibration of models.
- Dal Pozzolo et al. (2015); van den Goorbergh et al. (2022): resampling and
  weighting distort probabilities.
- Brier (1950): the score.

## Git habits used in this repo

- One logical change per commit, imperative subject under 72 characters.
- Results are committed only as the small report tables and figures; raw
  predictions and logs are ignored (`.gitignore`).
- Never commit `data/raw/`; the download script plus `data/checksums.json`
  reproduce it.
