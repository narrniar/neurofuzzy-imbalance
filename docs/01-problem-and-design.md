# Problem and experimental design

## The question

When the positive class is rare, does a gradient-trained TSK neuro-fuzzy
classifier give a better trade-off between

1. minority-class ranking (PR-AUC),
2. probability calibration (Brier, ECE, slope, intercept), and
3. model size (rules, parameters)

than LightGBM, logistic regression and a small MLP, under one shared protocol?

Sub-questions:

- **SQ1** How do the differences change from the natural rate to 5% to 1%?
- **SQ2** How much do learned membership functions and the rule count matter?
- **SQ3** Does class weighting help ranking but hurt calibration, for every model?

## Why these three properties

A rare-class model is used either to rank cases for manual review (ranking) or
to compare a probability with a cost threshold (calibration). Resampling and
class weighting are known to distort probabilities (Dal Pozzolo et al. 2015; van
den Goorbergh et al. 2022), and boosted trees and neural networks are known to
be miscalibrated in different ways (Niculescu-Mizil & Caruana 2005; Guo et al.
2017). A model that ranks well but outputs probabilities that are off by a factor
of ten is only useful with post-hoc correction, so the protocol measures both.

## Datasets and settings

| Dataset | Rows | Raw features | Natural rate | Split |
|---|---|---|---|---|
| Default of Credit Card Clients (UCI 350) | 30,000 | 23 | 22.1% | stratified |
| Bank Marketing, bank-additional-full (UCI 222) | 41,188 | 19 (duration dropped) | 11.3% | stratified |
| Credit Card Fraud (ULB, OpenML 1597) | 284,807 | 29 (Time used for ordering only) | 0.17% | by time |

The first two are thinned to 5% and 1% by removing positives at random before the
split; the fraud data stays at its natural rate. That gives seven settings. At 1%
the test split holds only 47 (credit) and 74 (bank) positives, so those results
are noisy by construction; the report prints the spread over seeds.

Split: 60% train, 10% validation (tuning, early stopping), 10% calibration (Platt
scaling, threshold), 20% test (used once per final model), 5 seeds.

Leakage controls: `duration` removed from Bank Marketing (known only after the
call), every fitted preprocessing step (one-hot vocabulary, scaler, feature
ranking, k-means for rule centres) fitted on the training split only, fraud split
by time so no model trains on later transactions, duplicate rows between train
and test counted and reported.

## Models

| Model | Role | Hyperparameters searched |
|---|---|---|
| TSK (HTSK) | model under test | rules 4/8/16, sigma scale, learning rate, weight decay, batch size, UR weight |
| LightGBM | strongest expected baseline | learning rate, leaves, min child samples, subsample, colsample, lambda |
| Logistic regression (L2) | simple, usually well calibrated | C |
| MLP | same parameter budget as TSK, tests whether the fuzzy structure matters | rules-equivalent, layers 1/2, dropout, learning rate, weight decay, batch size |

Every model gets 20 random configurations per setting, chosen by validation
PR-AUC, then the winner is retrained on 5 seeds. Class weighting is a separate
switch: `pos_weight` in the BCE loss (TSK, MLP), `scale_pos_weight` (LightGBM),
`class_weight="balanced"` (logistic regression), all set to the negative-to-
positive ratio of the training split.

## TSK ablations (SQ2)

Starting from the selected unweighted TSK configuration: fixed versus learned
membership functions, 4/8/16 rules, uniform regularisation on/off, and the
top-10 features instead of all features (the feature subset is applied to every
model so the comparison stays fair).

## Metrics

- Ranking: PR-AUC (primary), ROC-AUC (for comparison with other papers).
- Calibration: Brier score, log loss, ECE with 15 equal-size bins, calibration
  slope and intercept, reliability plots; all repeated after Platt scaling.
- Decision level: precision, recall, F1 at the F1-maximising threshold chosen on
  the calibration split.
- Size: rules, antecedents per rule, parameters; LightGBM trees and leaves.
- Resources: fit time, prediction time per 1,000 rows, process peak memory.

## Reading the result (section 6.4 of the proposal)

- Supports the claim: TSK within 0.02 PR-AUC of the best baseline and either
  better calibrated (lower ECE or Brier) or smaller.
- Qualifies: the advantage appears only in some settings or only before Platt.
- Contradicts: TSK worse than both LightGBM and logistic regression on ranking and
  calibration.

The report applies these rules mechanically per setting (`verdict.csv`), and the
interpretation in `docs/04-results-and-interpretation.md` adds the judgement.
