# Presentation outline and demonstration

1. **Title and team:** Predictive Box Office Revenue and Movie Success Analytics.
2. **Problem:** worldwide revenue regression plus a transparent binary success proxy.
3. **Data:** 4,803 movies and matching credits; 3,216 eligible records after documented filtering.
4. **Preparation:** JSON parsing, invalid finances, encoding, and shared preprocessing pipelines.
5. **EDA:** budget/revenue, skewed revenue, genre patterns; association does not imply causation.
6. **Feature design:** pre-release-style fields; exclude popularity, ratings, votes, and revenue-derived features.
7. **Experimental design:** temporal train/validation/test, fixed parameters, baseline comparisons.
8. **Regression:** show validation comparison and held-out plots. MAE $93.5M versus $155.6M baseline.
9. **Classification:** majority baseline wins F1; AUC 0.50 means no discrimination. Explain why F1 alone can mislead.
10. **Demo and future work:** show one scenario, filtered movies, results, limitations, and next research steps.

## Three-minute demo

- Open Overview and state sample size and historical coverage.
- Open Predict a movie; use a $30M budget, Drama, 110 minutes, and June 2015.
- Submit; explain that revenue is individualized while selected success probability is constant.
- Change budget to $60M and compare revenue; do not claim causal effects.
- Explore a genre/year filter and download rows.
- Show test results, the baseline comparison, confusion matrix, and limitations.

## Questions to prepare for

- Why regression and classification? Different targets: a continuous gross and a defined binary proxy.
- Why chronological splitting? It better simulates future-release prediction than random mixing.
- Why exclude ratings? Their retrospective values would reveal post-release information.
- Why log revenue? To handle skew; it can trade blockbuster dollar accuracy for typical-film fit.
- Is 2x budget profitability? No; marketing and revenue sharing are absent.
- Why is classifier F1 high but AUC 0.5? It always predicts success, achieving high recall without ranking movies.
- Why no deep learning? Small tabular data and a one-day scope favor reproducible traditional models.
- Can this predict a 2026 blockbuster reliably? That has not been established on this historical sample.
