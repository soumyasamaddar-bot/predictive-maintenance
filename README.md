# Predictive Maintenance of Industrial Equipment using IoT Sensor Data

An end-to-end, script-based analytics pipeline that predicts machine failure
from IoT sensor readings. It covers data cleaning, EDA, feature engineering,
model training/evaluation, machine risk scoring and export of dashboard-ready
tables for Power BI / Tableau.

> **Standards:** pure Python `.py` files only - no notebooks, no Colab. All
> paths, thresholds, column names and hyper-parameters live in `config.py`.

**GitHub repository:** https://github.com/soumyasamaddar-bot/predictive-maintenance

## Project structure

```text
predictive_maintenance/
├── main.py                  # Entry point / pipeline orchestration
├── app.py                   # Streamlit front end (upload a CSV and analyse it)
├── assets/cronus_logo.png   # Logo shown in the app
├── config.py                # All configuration & constants
├── requirements.txt
├── data/
│   ├── raw/                 # sensor_data.csv (input)
│   └── processed/           # cleaned + feature datasets
├── src/
│   ├── data_generator.py    # Synthetic IoT data (used only if no raw data supplied)
│   ├── data_loader.py       # Load / validate / save CSV
│   ├── preprocessing.py     # Cleaning (duplicates, units, invalid values, gaps)
│   ├── feature_engineering.py  # Rolling, lag, deviation, band & machine features
│   ├── analysis.py          # EDA tables, anomaly detection (no plotting)
│   ├── modeling.py          # Chronological split, pipelines, training, persistence
│   ├── evaluation.py        # Metrics, feature importance, threshold analysis
│   ├── reporting.py         # Risk scoring, KPIs, recommendations, exports
│   ├── app_service.py       # Upload validation + in-memory analysis for the app
│   ├── visualization.py     # All charts (no business logic)
│   └── utils.py             # Logging + directory helpers
├── tests/test_pipeline.py   # Unit tests
└── outputs/
    ├── figures/             # 16 PNG charts
    ├── models/              # Saved model (joblib)
    ├── reports/             # EDA/model tables + run_summary.txt
    ├── dashboard/           # Power BI / Tableau data sources
    └── logs/pipeline.log
```

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                     # full pipeline
python main.py --regenerate-data   # rebuild the synthetic dataset first
python main.py --skip-plots        # faster, no figures
python -m unittest discover -s tests   # run tests
```

**Using your own dataset:** place a CSV at `data/raw/sensor_data.csv` with the
columns `Machine_ID, Timestamp, Temperature, Pressure, Vibration, RPM, Current,
Voltage, Operating_Hours, Maintenance_Count, Machine_Age, Failure`. If the file
is absent, a realistic synthetic dataset (20 machines, ~29k readings, with
injected missing values, duplicates, outliers and unit errors) is generated.
Column names, units and valid ranges can be adapted in `config.py`.

## Streamlit app (upload your own dataset)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the address shown in the terminal (usually http://localhost:8501), then:

1. In the sidebar choose **Upload a CSV file** (a template CSV can be downloaded there) or
   **Use the sample dataset**.
2. Press **Run analysis**. Files with missing columns, too few rows or no failure labels
   show a clear error message instead of crashing.
3. Explore the tabs: **Overview**, **Data quality**, **Exploration**, **Model** (with an
   alert-threshold slider), **Machine risk** (priority list and recommendations) and
   **Downloads** (CSV files usable as Power BI data sources).

The app uses the same modules as `main.py` (`src/app_service.py`) and keeps everything in
memory, so uploads never overwrite the project's own outputs.

## Pipeline

| Stage | Module | What happens |
| --- | --- | --- |
| 1 Load | `data_loader` | Read CSV, validate schema, clear errors on failure |
| 2 Clean | `preprocessing` | Types, duplicates, deg F to deg C, invalid values to NaN, past-only forward fill |
| 3 EDA | `analysis`, `visualization` | Failure by machine, sensor vs failure, correlations, trends, outliers, Isolation Forest anomaly scores |
| 4 Features | `feature_engineering` | Rolling mean/max/min/std, lags, deviation from normal range, hour/age bands, maintenance frequency, machine-level stats |
| 5 Model | `modeling`, `evaluation` | Logistic Regression (baseline), Decision Tree, **Random Forest (primary)**, Gradient Boosting |
| 6 Score | `reporting` | Failure probability, Healthy / Warning / High Risk, ranked maintenance list |

### Leakage prevention
- **Chronological split**: the last 25% of time is the test set (no random shuffling).
- **Embargo**: training rows within 24 h of the split are dropped because their labels look into the test period.
- **Causal features**: rolling, lag and expanding features use past readings only; gap-filling is forward-only.
- **Train-only statistics**: imputation and scaling are fitted inside the model pipeline on training data only.

### Evaluation
Confusion matrix, accuracy, precision, recall, F1, ROC-AUC, PR-AUC, feature
importance and a threshold table. Recall is prioritised: a missed failure
(false negative) means unplanned downtime, while a false alarm (false positive)
only costs an inspection.

## Outputs for the Power BI / Tableau dashboard (`outputs/dashboard/`)

| File | Use |
| --- | --- |
| `kpi_summary.csv` | KPI cards (total / healthy / warning / high-risk machines, failure rate, avg temperature and vibration) |
| `machine_risk_summary.csv` | Machine-wise risk, probability, priority rank, recommendation |
| `scored_readings.csv` | All readings with predicted probability, for trends and filters |

## Web dashboard
A single-page web dashboard (`index.html`) shows the latest run: fleet status board,
alert-threshold slider, model comparison, top features and weekly trend.
Live site: https://soumyasamaddar-bot.github.io/predictive-maintenance/  (works after enabling GitHub Pages in the repository settings)
It shows saved results from the pipeline and does not score new data live.

## Power BI
`project_dashboard.pbix` (4 pages: Executive Overview, Sensor Analysis, Machine Health,
Predictive Maintenance) is built from the CSVs in `outputs/dashboard/` and `outputs/reports/`.

## Important notes
- With the **synthetic** dataset, sensors drift clearly before failure, so the
  scores are very high. Expect lower scores on real data; **report results
  honestly and discuss this limitation** in your documentation.
- Risk thresholds (`RISK_THRESHOLDS`), the decision threshold and the prediction
  horizon are business choices - tune them in `config.py`.
