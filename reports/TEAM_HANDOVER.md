# Five-member review and presentation handover

These are suggested ownership areas for reviewing and understanding the completed project.
They are not claims about who authored the implementation. Replace member labels with names.

| Member | Review area | Evidence to understand | Presentation responsibility |
| --- | --- | --- | --- |
| 1 | Data and cleaning | src/data.py, data audit, excluded finances | Data provenance and quality |
| 2 | EDA and statistics | EDA charts, genre summary, correlations | Findings and caveats |
| 3 | Revenue modeling | Regressors, log target, MAE/RMSE/R², residuals | Model comparison and errors |
| 4 | Classification | F1, AUC, baseline behavior, confusion matrix | Explain the negative classification finding |
| 5 | Application and integration | app.py, README, verification checks | Run demo and explain deployment |

Each member should run the app, inspect the notebook, and explain the temporal split and leakage rules.
Before submission, add the institution's required cover details, team names/IDs, and any required
AI-assistance disclosure. The report and presentation outline contain actual results, not placeholders.
