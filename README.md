# Predictive Box Office Revenue and Movie Success Analytics

A complete data science project using the supplied TMDB 5000 movie and credits datasets.
It includes reproducible cleaning, exploratory analysis, temporal model comparison,
saved regression/classification pipelines, historical evaluation, and a Streamlit demo.

**Current revision: v2.** Both deployed estimators are Random Forest models. Success
probabilities now depend on movie inputs. See [the classifier revision and viva notes](reports/V2_CLASSIFICATION_VIVA.md).

## Start the demo on Windows

Install Python 3.13, extract the complete project folder, then double-click `run_app.bat`.
The first run installs dependencies; internet access is required for setup. Trained models
are already included. Keep the terminal open while using the app; press Ctrl+C to stop.

Manual setup, from this folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the localhost URL printed by Streamlit (usually http://localhost:8501).
The lock file records the tested environment. `requirements.txt` lists direct dependencies.
Use the tested versions to load the supplied scikit-learn model artifacts.

## Reproduce training and checks

```powershell
.venv\Scripts\python.exe -m src.train
.venv\Scripts\python.exe verify_project.py
```

Training regenerates processed data, figures, validation comparisons, final test predictions,
feature importance, audit information, and model artifacts. It may take several minutes.
The notebook in `notebooks/` walks through the same workflow and displays saved results.

## Research questions

1. How well can pre-release-style metadata predict worldwide box office revenue?
2. Can the same information classify movies whose revenue reaches twice their budget?
3. Which features contribute to predictive performance, and where do the models fail?

## Method

- Join unique `id` and `movie_id` keys; preserve original CSV files.
- Require positive budget and revenue, a valid release date, and budget at least $1,000.
  The last condition excludes suspicious values, not verified errors. This limits generalization
  to movies with reported financial data; excluded zero revenues must not be labeled failures.
- Features: log budget, runtime, release year, cyclical release month, language,
  first-listed production company, director, and 20 multi-label genre indicators.
  Cast is parsed for future use but is not a model input.
- Train through 2010, validate on 2011–2013, test on 2014–2016.
- Impute/scale numeric inputs and encode infrequent/unknown categories inside each pipeline.
  Vocabulary/frequency decisions and imputation are fitted only on training data.
- Compare median baseline, Ridge, Random Forest, and Gradient Boosting regression.
  Non-baseline regressors train on log1p revenue; predictions invert with expm1.
- Compare majority baseline, Logistic Regression, and Random Forest classification.
- Select regression by lowest validation dollar MAE and classification by highest validation ROC-AUC
  (lowest Brier score breaks an AUC tie). Training fails rather than publishing a constant/chance classifier as individualized predictions.
  Hyperparameters are fixed in the source; no extensive search or test-driven tuning is claimed.
- Refit winners on train + validation, then evaluate on the historical 2014–2016 period.
  That period was already evaluated in v1; v2 results are a reused historical comparison,
  not a new untouched external test. Selection still uses validation only.
- Report MAE/RMSE/R² and accuracy/precision/recall/F1/macro-F1/balanced accuracy/ROC-AUC/Brier score.
- Permutation importance uses validation data and the original train-only fitted model.
  It describes global predictive associations, not causal effects or individual explanations.

Success is `revenue >= 2 * budget`. It is a commercial-success proxy, not audited profitability.
Classifier threshold is fixed at 0.50; probabilities are not separately calibrated.
Separate revenue and success models can produce different conclusions. The app displays
  the point-revenue threshold comparison independently and explicitly flags disagreement.

## Files

| Path | Purpose |
|---|---|
| `app.py` | Five-section interactive application |
| `src/data.py` | Shared cleaning and inference feature contract |
| `src/train.py` | End-to-end training, selection, evaluation, figures |
| `data/raw/` | Unmodified supplied movies/credits CSVs |
| `data/processed/movies_clean.csv` | Auditable cleaned data with split labels |
| `models/` | Fitted pipelines and app input options |
| `reports/metrics.json` | Selected-model and baseline final test metrics |
| `reports/*_validation.csv` | Candidate comparisons on validation only |
| `reports/test_predictions.csv` | Individual held-out predictions |
| `reports/figures/` | EDA and evaluation charts |
| `reports/PROJECT_REPORT.md` | Findings, methodology, actual results, limitations |
| `reports/PRESENTATION_OUTLINE.md` | Suggested slide content and demo sequence |
| `reports/TEAM_HANDOVER.md` | Suggested five-member review/presentation roles |
| `verify_project.py` | Integration and app checks |

## Limits and responsible interpretation

This is a historical educational study, not a production forecasting service. The sample
is not representative of every release; reported budgets and revenues can be incomplete
or inaccurate. Monetary values are treated as nominal USD and are not inflation-adjusted.
The eligible snapshot ends during 2016, and final deployed models train through 2013.
Newer-year inputs extrapolate beyond training coverage. Marketing, theater count,
competition, audience demand, and distribution strategy are absent. Metadata was collected
after release; although selected fields plausibly exist pre-release, their historical
availability and unchanged values were not independently verified.

No revenue prediction interval or calibrated confidence claim is made. Log-target models
may underestimate blockbuster dollar revenue; inspect error plots and baseline comparisons.
Do not interpret correlation or importance as evidence that changing a feature causes success.

## Data provenance

Source: [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
The two raw CSV files were provided by the project owner. Their SHA-256 checksums are in
`reports/data_audit.json`. Metadata originates from The Movie Database (TMDB).
Keep attribution and consult the original source terms before redistributing source data.
This project is not endorsed by TMDB. No third-party project code or trained models were used.
