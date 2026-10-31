# Presentation outline for revision v2

1. Problem: revenue and 2x-budget success proxy; not audited profitability.
2. Data: 4,803 raw, 3,216 eligible; financial exclusions and their bias.
3. Features: 28 inputs; no ratings/popularity/revenue leakage.
4. Chronology: train through 2010; validate 2011–2013; refit through 2013.
5. Revenue: validation MAE selects Random Forest; historical MAE $93.55M vs $155.59M baseline.
6. Original classifier defect: positive F1 selected a constant prior.
7. V2 correction: select by validation ROC-AUC; Random Forest wins .615.
8. Historical classifier results: accuracy 64.3%, AUC 0.669; modest discrimination.
9. Demo: use the below/above-50 examples in V2_CLASSIFICATION_VIVA.md; show both independent outcomes.
10. Limits: uncalibrated probabilities, old/selected data and reused evaluation period; need fresh data.

Do not claim v2 was evaluated on a newly untouched external test. Do not present the old
24-page guide's constant-classifier result as the current implementation.
