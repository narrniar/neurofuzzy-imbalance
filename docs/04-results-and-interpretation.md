# Results and interpretation (full protocol, 2026-11-04)

1,680 fits, 7 settings x 4 models x 2 weightings x 20 configurations, 5 seeds,
TSK ablations, zero failed fits, 15 minutes on an M4 Pro. Full tables:
`results/report/full/REPORT.md`.

## Ranking (PR-AUC, no class weighting, mean over 5 seeds)

| setting | TSK | LightGBM | LogReg | MLP |
|---|---|---|---|---|
| credit default, natural (22%) | 0.544 | **0.555** | 0.500 | 0.530 |
| credit default, 5% | 0.229 | **0.232** | 0.203 | 0.206 |
| credit default, 1% | 0.049 | **0.057** | 0.047 | 0.047 |
| bank marketing, natural (11%) | 0.447 | **0.468** | 0.445 | 0.452 |
| bank marketing, 5% | 0.318 | **0.331** | 0.313 | 0.316 |
| bank marketing, 1% | 0.092 | **0.106** | 0.098 | 0.088 |
| fraud, natural (0.17%) | 0.777 | 0.680 | 0.775 | **0.803** |

Paired differences (TSK minus baseline, 95% bootstrap CI over test rows, pooled
over seeds):

| setting | vs LightGBM | vs LogReg |
|---|---|---|
| credit natural | -0.010 [-0.015, -0.004] | +0.045 [+0.036, +0.052] |
| credit 5% | -0.003 [-0.019, +0.010] | +0.026 [+0.014, +0.040] |
| credit 1% | -0.008 [-0.021, +0.003] | +0.002 [-0.010, +0.012] |
| bank natural | -0.022 [-0.028, -0.015] | +0.002 [-0.003, +0.007] |
| bank 5% | -0.013 [-0.022, -0.005] | +0.005 [-0.004, +0.014] |
| bank 1% | -0.015 [-0.034, +0.001] | -0.006 [-0.023, +0.007] |
| fraud | +0.096 [+0.056, +0.136] | +0.002 [-0.023, +0.030] |

## Calibration (no class weighting)

| setting | metric | TSK | LightGBM | LogReg | MLP |
|---|---|---|---|---|---|
| credit natural | ECE | 0.021 | 0.018 | 0.059 | 0.020 |
| credit 5% | ECE | 0.009 | 0.020 | 0.019 | 0.011 |
| credit 1% | ECE | 0.005 | 0.005 | 0.006 | 0.005 |
| bank natural | ECE | 0.013 | 0.010 | 0.013 | 0.013 |
| bank 5% | ECE | 0.011 | 0.011 | 0.008 | 0.012 |
| bank 1% | ECE | 0.031 ± 0.017 | 0.006 | 0.004 | 0.005 |
| bank 1% | intercept | -1.32 ± 0.54 | +0.06 | +0.01 | +0.05 |

After Platt scaling every model lands at ECE 0.003 to 0.02 in every setting; the
only one that does not fully recover is logistic regression on credit default
(0.06, a slope problem, not an intercept problem).

## What this says about the research question

**Expected outcome 1 (LightGBM ranks best) holds in 6 of 7 settings.** The TSK is
the second-best ranker on credit default and within 0.02 PR-AUC of LightGBM in 5
of 7 settings. Where the CI excludes zero against TSK, the gap is 0.01 to 0.02.

**The TSK beats logistic regression only where the signal is non-linear.** On
credit default the gain is +0.045 PR-AUC with a tight CI; on bank marketing and
fraud the two are indistinguishable. This is the clearest statement the data
supports about what fuzzy rules add: a non-linear boost over a linear model,
not an edge over trees.

**Expected outcome 2 (TSK better calibrated than the MLP) does not hold.** Raw
ECE is the same for TSK and MLP within noise in every setting, and at bank 1% the
TSK is the worst-calibrated model, with a large seed variance in its intercept.
That one setting is the only "contradicts" verdict.

**Expected outcome 3 (class weighting hurts calibration for every model) holds
exactly.** Weighted minus unweighted, mean over seeds:

| | d PR-AUC | d ECE | d intercept |
|---|---|---|---|
| TSK | -0.02 to +0.00 | +0.18 to +0.36 | -1.2 to -5.7 |
| LightGBM | -0.06 to -0.01 | +0.01 to +0.23 | -0.4 to -4.8 |
| LogReg | -0.01 to +0.00 | +0.07 to +0.40 | -1.2 to -8.5 |
| MLP | -0.04 to +0.00 | +0.09 to +0.37 | -1.3 to -8.1 |

Ranking is flat (LightGBM even loses), the intercept drops by roughly the log of
the weight ratio, and Platt scaling repairs all of it. Weighting buys nothing
here except a different default threshold.

**Model size (SQ2):** selected TSKs have 400 to 3,000 parameters, usually more
than the matched MLP and far fewer than LightGBM's leaf count. Learned membership
functions matter on credit default (fixed MFs: -0.036 PR-AUC at the natural
rate, -0.017 at 5%); 4 rules are as good as 16 on bank marketing and credit at
1%, and better on fraud (0.803 vs 0.777). Uniform regularisation changes nothing
measurable. The top-10 feature subset costs every model 0.01 to 0.03 PR-AUC on
credit default and nothing on fraud.

**Subgroups (credit default, natural rate):** every model has a higher false
negative rate for the 30-39 age group (0.49 to 0.54) than for 50+ (0.41 to 0.43)
and flags men more often than women (0.25 vs 0.20). The pattern is identical
across models, so it is a property of the data and the threshold, not of the TSK.

## Two protocol weaknesses to fix before writing this up

1. **Early stopping on a 24-positive validation split.** LightGBM's fraud result
   (0.680) is an artefact: validation PR-AUC on 24 frauds is noise, and early
   stopping chose 2 and 7 trees on two of the five seeds. The same problem
   affects the PyTorch models less because they stop on epochs, not trees, but
   it is still there. Fix: stop on validation log loss, or pick the number of
   trees on validation with a smoother criterion. Expect LightGBM on fraud to
   move to about 0.80 PR-AUC, in line with published numbers.

2. **Configuration selection on a single seed.** The TSK configuration chosen on
   seed 0 for fraud (16 rules) scores 0.777 ± 0.028 over seeds, while the 4-rule
   ablation of the same configuration scores 0.803 ± 0.003. Selecting on the
   mean of two or three seeds would avoid picking seed-lucky configurations for
   every model.

Both are a few lines in `search.py` / `experiment.py` and a 15-minute rerun. The
report should be regenerated and this page updated afterwards.

## Verdict per setting (section 6.4 rules applied mechanically)

supports: credit 5% and 1% (unweighted), credit natural and 1% (weighted), bank
natural and 5% (one weighting each); qualifies: 7 settings; contradicts: bank 1%
unweighted. The honest one-line summary: the TSK is a competent second-tier
ranker with no calibration or size advantage, and its claimed benefits do not
show up as a trade-off advantage over LightGBM as the positive class becomes
rarer.
