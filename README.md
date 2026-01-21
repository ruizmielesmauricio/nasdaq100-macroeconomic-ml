# nasdaq100-macroeconomic-ml

### Project Summary

End-to-end machine learning pipeline that models and predicts next-day NASDAQ-100 market direction and returns using:
	•	Top NASDAQ-100 constituents (Yahoo Finance)
	•	Macroeconomic indicators from FRED
	•	XGBoost regression and classification models

The project emphasises reproducibility, chronological validation, and model interpretability, and is designed to run both locally and via GitHub Actions.

⸻

### Data Sources
	•	Equity data: Yahoo Finance (Top NASDAQ-100 constituents)
	•	Index proxy: NASDAQ-100
	•	Macroeconomic indicators:
	•	Federal Funds Rate
	•	CPI
	•	Unemployment
	•	GDP
	•	10-Year Treasury Yield
	•	M2 Money Supply

### Project Structure

nasdaq100-macroeconomic-ml/
├── src/
│   └── main.py                # Full ML pipeline
├── data/                      # Generated datasets (local / Actions)
├── results/
│   └── figures/               # Saved plots (Actions artifacts)
├── .github/
│   └── workflows/
│       └── run_pipeline.yml   # GitHub Actions workflow
├── requirements.txt
└── README.md

### How to Run Locally

pip install -r requirements.txt
python src/main.py

This will:
	•	Download market and macroeconomic data
	•	Engineer features
	•	Train regression and classification models
	•	Save datasets and plots to data/ and results/figures/

⸻

Running via GitHub Actions (Recommended)

The pipeline is fully automated using GitHub Actions.

### How it works
	•	The workflow runs the entire pipeline on a clean environment
	•	Generated figures and evaluation metrics are uploaded as Artifacts
	•	No local setup is required

### How to access results
	1.	Go to Actions in the repository
	2.	Open a successful workflow run
	3.	Download the available Artifacts, which include:
	•	Model plots (results/figures)
	•	Evaluation metrics (e.g. classification_metrics.csv)

⸻

### Model Evaluation
	•	Regression: RMSE, R², prediction vs actual plots
	•	Classification:
	•	Precision
	•	Recall
	•	F1 Score (primary metric)
	•	Confusion matrix
	•	Feature importance

The F1 score is used as the primary classification metric to balance precision and recall in a noisy financial prediction setting.

