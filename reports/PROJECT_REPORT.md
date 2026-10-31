# Movie revenue and success analytics current project report

Revision v2, 31 October 2026. This report supersedes the v1 baseline-selection results.
Original v1 results are retained under `reports/archive/` for comparison.

## Problem and data

Predict worldwide box office revenue from pre-release-style movie metadata and estimate
whether revenue reaches at least twice production budget. This success label is a proxy,
not audited profit. The supplied TMDB movie and credits files each contain 4,803 rows,
matched one-to-one by movie ID. Raw inputs are unchanged and checksummed in `data_audit.json`.

Require positive budget and revenue, a valid release date, and budget of at least $1,000.
Zero financial values may mean unreported values; do not call all of them failures. The
budget cutoff excludes 13 suspicious records and is an explicit assumption. There are
3,216 eligible movies: 1,803 success cases and
1,413 below-threshold cases. Financial filtering introduces selection bias.

## Features and preparation

`src/data.py` parses genres and credits, extracts the first-listed company and director,
and constructs 28 named inputs: log budget, runtime, release year, cyclical month coordinates,
language, company, director, and 20 multi-label genre flags. Cast is parsed but not modeled.
Actual revenue, success, popularity, ratings, votes and all revenue-derived fields are excluded.

`src/train.py` fits median imputation and StandardScaler on numeric columns, one-hot encoding
with infrequent grouping (minimum count 8) for categories, and unchanged genre indicators.
These transformations remain inside each saved pipeline. Unknown categories are supported.
The app calls the same `make_features` function as training.

## Experimental design

Training uses 2,433 movies through 2010; validation uses 455 releases from 2011–2013.
The selected pipelines refit on 2,888 train-plus-validation movies. Evaluation uses 328
releases from 2014–2016. **This historical evaluation period was inspected in v1. It is
not a new untouched external test for v2.** Model selection in v2 uses validation results
only; no additional candidates or hyperparameters were chosen from v2 evaluation outcomes.
New research should obtain a fresh final period.

Revenue candidates are median baseline, Ridge (alpha 10), Random Forest (180 trees, depth 16,
minimum leaf 3), and Gradient Boosting (180 stages, rate .04, depth 2, minimum leaf 10, Huber).
Learned regressors fit log1p revenue and invert with expm1. Revenue selection still minimizes
validation dollar MAE. Hyperparameters are fixed and seed is 42, not an exhaustive search.

Success candidates remain majority prior baseline, Logistic Regression (C .3, max_iter 2500),
and Random Forest (180 trees, depth 12, minimum leaf 5, seed 42). V1 selected by positive-class
F1, which favored always predicting the common positive class. V2 instead selects by
**validation ROC-AUC**, with lower Brier score as the tiebreak. This matches the requirement
for movie-specific probability ranking. The baseline is a comparison, never silently a personalized
predictor: training stops if the winner is constant or validation AUC is no better than .50.
The decision cutoff stays .50; no artificial probability offsets or forced failures are applied.

## Selection evidence

Random Forest classification has validation ROC-AUC 0.615 versus 0.590 for Logistic Regression
and 0.500 for the prior baseline. Its validation positive-class F1 remains lower than the
baseline's; the metric change reflects a different product objective rather than a claim
that every score improved. Full comparisons are in the validation CSVs.

Both deployed models are now **Random Forest**. The classifier uses movie features and
returns probabilities on both sides of .50. Input-option lists now use development records
only. No separate probability calibration was applied; varying probability is not proof of calibration.

## Historical evaluation results

| Classification metric | Random Forest v2 | Prior baseline |
| --- | ---: | ---: |
| Accuracy | 0.6433 | 0.6220 |
| Precision | 0.6835 | 0.6220 |
| Recall | 0.7941 | 1.0000 |
| Positive-class F1 | 0.7347 | 0.7669 |
| Macro-F1 | 0.5953 | 0.3835 |
| Balanced accuracy | 0.5946 | 0.5000 |
| ROC-AUC | 0.6688 | 0.5000 |
| Brier score | 0.2260 | 0.2398 |

The revised classifier has modest discrimination, with lower probability error than the
baseline on this historical comparison. Positive-class F1 decreases because it no longer
predicts every movie as successful. It improves macro-F1 and balanced accuracy. Revenue
results remain approximately MAE $93,546,203, RMSE $175,212,067, and R² 0.523.
The median revenue baseline MAE is $155.59M; the forest reduces MAE by about 39.9% relative to that baseline.

![Success confusion matrix](figures/confusion_matrix.png)
![ROC curve](figures/roc_curve.png)

## Interpretation and examples

Permutation importance uses the original train-only fitted model on validation records.
Success importance now uses ROC-AUC rather than positive-class F1; revenue still uses MAE.
Importance measures associations, not causes or a personalized explanation.

`prediction_examples.json` stores hypothetical scenarios demonstrating probabilities below
and above .50. They are selected from a fixed input grid, not real films with validated
future outcomes. For example, $15M budget, 90 minutes, Drama, English, June 2013, Unknown
company and director gives approximately 43.1%; $1M, 140 minutes, Horror with the same
date/language/unknown identities gives approximately 60.5%.

The app labels the independent classifier decision, the below-threshold probability, and
the separate comparison of point-estimated revenue against 2x budget. It warns if the two
models disagree. A point estimate below a threshold does not mathematically require a
separately fitted classifier probability below .50.

## Software tests and operation

Run `python -m src.train` to regenerate results; run `python verify_project.py` to verify
temporal isolation, target rules, saved-model/export agreement, unknown inputs, nonconstant
probabilities, both sides of .50, validation AUC selection, all app pages, and the actual
form with below/above-threshold examples. Model/data cache keys change with this revision.
The app loads saved pipelines rather than retraining per request.

## Limits

The sample is selected and historical, money is nominal without inflation adjustment,
and financial reporting may be incomplete. Pre-release availability of retrospective fields
was not independently timestamp-verified. Marketing, theaters, audience demand and competition
are absent. Probability calibration is unverified, and .669 AUC is not strong production
forecasting evidence. The reused evaluation set must not be called a fresh independent test.

## Sources

- Project: https://github.com/krishPatel200/movie-success-analytics
- Data: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- Metrics: `metrics.json`; selection: `classification_validation.csv`.
- AUC: https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.roc_auc_score.html
- Probability calibration: https://scikit-learn.org/1.7/modules/calibration.html
