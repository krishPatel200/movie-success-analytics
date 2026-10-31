# Current classifier and viva corrections for revision v2

Updated 31 October 2026. **Read this before using the older 24-page Word viva guide.**
That document explains v1, which deployed a constant prior. The current implementation no
longer does so. Its dataset, regression, and general ML explanations remain useful, but
the following corrections supersede its classification results and model-selection sections.

## What changed and why

V1 selected a DummyClassifier because positive-class F1 favored always predicting success.
That is unsuitable for an interface promising movie-specific probability. V2 ranks the same
candidate classifiers by validation ROC-AUC, with Brier as a tiebreak. Random Forest wins
validation AUC .615 and is refitted on train + validation. The baseline remains visible as
a comparator. No probabilities are manually reduced to manufacture failures.

## Updated short answers

- **Which models are deployed?** Random Forest Regressor and Random Forest Classifier.
- **Where is the code?** `src/train.py`: classifiers dictionary and validation sorting;
  `app.py`: Predict a movie; `models/success_pipeline.joblib`: the new fitted classifier.
- **Does the success percentage depend on inputs?** Yes. Budget, runtime, genre, release
  timing, language, company and director feed a fitted preprocessing/classifier pipeline.
- **Is success actual profit?** No. The label is reported revenue >= twice production budget.
- **What is the cutoff?** Probability .50; it has not been adjusted just to create failures.
- **What is the classifier performance?** Historical accuracy 64.3%, AUC 0.669,
  F1 0.735, macro-F1 0.595, balanced accuracy 0.595.
- **Why is F1 lower than the prior baseline?** The objective now rewards discrimination
  rather than only positive-class recall. Always predicting success had high recall but AUC .50.
- **Is probability calibrated?** No separate calibration was performed. It is a model estimate.
- **Can the revenue and success outputs disagree?** Yes; they are separate models. The app
  now shows both conclusions explicitly and highlights disagreement.
- **Is the evaluation independent?** This is the same 2014–2016 historical period evaluated
  in v1, not a new untouched test. V2 selection uses 2011–2013 validation only. Fresh data is
  required before claiming independent current-market accuracy.
- **Did you use deep learning or force below-50 results?** No. The change is a validation
  metric and trained-model selection correction using the existing candidates.

## Reproducible demo examples

Use English (`en`), release date 1 June 2013, company **Unknown**, director **Unknown**.

| Budget | Runtime | Genre | Estimated success probability |
| --- | ---: | --- | ---: |
| $15,000,000 | 90 minutes | Drama | 43.1% |
| $1,000,000 | 140 minutes | Horror | 60.5% |

These are hypothetical input combinations, not proof about real future movie outcomes.
The exact model-generated values are in `prediction_examples.json`. A UI integration test
submits both scenarios and checks that their displayed values cross .50.

## Team study assignments

1. Data preparation: original guide pages 2–4; cleaning and targets are unchanged.
2. Features/EDA: original pages 4–7; dropdown categories now come from development data only.
3. Revenue models: original pages 8–10 and 12; revenue approach is unchanged.
4. Classification: this note plus current PROJECT_REPORT.md; old pages 8–11 and Q42–45/Q53/Q59
   describe the v1 problem and must not be presented as the current behavior.
5. App/testing: current app.py and verify_project.py; demonstrate both scenarios and separate
   revenue-based outcome, then explain the reused historical evaluation caveat.

## Current authoritative files

Read `PROJECT_REPORT.md`, `classification_validation.csv`, `metrics.json`, and this note.
V1 result snapshots are in `archive/`. The earlier Word guide is historical where it states
the deployed classifier always returns 55.37% or has AUC .50.
