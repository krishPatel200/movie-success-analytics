# Deploy on Streamlit Community Cloud

1. Push this folder's contents to a GitHub repository. `app.py` must be at its root.
2. Sign in at https://share.streamlit.io and choose **Create app**.
3. Select the repository and branch `main`, with main file path `app.py`.
4. In advanced settings choose Python **3.13**, matching the model-training environment.
5. Deploy and wait for dependency installation and application startup.

No API keys, database, or secrets are required. Models are already trained and committed.
The pinned `requirements.txt` is suitable for the Linux hosting environment; the full
`requirements-lock.txt` records the original Windows development environment and is used
by the Windows launcher. Do not substitute that Windows lock file for cloud dependencies.

If model loading fails, confirm scikit-learn is 1.7.2. If files are missing, check that
`models/`, `data/processed/`, and `reports/` were committed. The raw data is required only
for retraining/verification, not normal app startup. Choose **Reboot app** after fixing
deployment configuration. Do not retrain on each app startup.

After deployment, verify Overview, Predict a movie, Explore the data, Model evaluation,
and Methodology. Confirm the classification baseline disclosure is visible.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
