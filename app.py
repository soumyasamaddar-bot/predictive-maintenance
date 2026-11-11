"""Streamlit front end for the Predictive Maintenance project.

Run with:
    streamlit run app.py

Users upload a sensor CSV (or use the bundled sample), then explore data
quality, exploratory analysis, model results and machine risk rankings.
All analysis is done by the same modules as ``main.py`` (src/app_service.py).
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import config
from src import app_service, data_loader, evaluation

LOGO = Path(__file__).parent / "assets" / "cronus_logo.png"
RISK_COLOURS = {"High Risk": "#D1431F", "Warning": "#C98A0A", "Healthy": "#2A8A74"}


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_analysis(file_bytes: bytes, compare_all: bool):
    """Run the analysis once per (file, option) pair. Returns (results, error)."""
    try:
        raw = app_service.read_uploaded_csv(io.BytesIO(file_bytes))
        return app_service.run_analysis(raw, compare_all), None
    except app_service.AnalysisError as exc:
        return None, str(exc)


def sample_bytes() -> bytes:
    """Bytes of the bundled synthetic dataset (generated if missing)."""
    raw = data_loader.load_raw_data(generate_if_missing=True)
    return app_service.to_csv_bytes(raw)


def template_bytes() -> bytes:
    """A 3-row example showing the expected columns."""
    return app_service.to_csv_bytes(pd.read_csv(io.BytesIO(sample_bytes()), nrows=3))


# --------------------------------------------------------------------------- #
# Tab renderers
# --------------------------------------------------------------------------- #
def render_overview(res: dict) -> None:
    ov, kpi = res["overview"], res["kpis"].set_index("KPI")["Value"]
    c = st.columns(4)
    c[0].metric("Machines", int(kpi["Total Machines"]))
    c[1].metric("Readings", f"{int(kpi['Total Readings']):,}")
    c[2].metric("Failure rate (labelled readings)", f"{kpi['Failure Rate (%)']:.2f}%")
    c[3].metric("High-risk machines", int(kpi["High Risk Machines"]))
    st.caption(f"Data covers {ov['start']:%d %b %Y} to {ov['end']:%d %b %Y}.")
    st.subheader("Data preview (cleaned)")
    st.dataframe(res["clean"].head(200))


def render_quality(res: dict) -> None:
    cl = res["cleaning"]
    st.subheader("Cleaning summary")
    c = st.columns(4)
    c[0].metric("Rows before / after", f"{cl['rows_before']:,} / {cl['rows_after']:,}")
    c[1].metric("Missing values fixed", f"{cl['missing_before']:,}")
    c[2].metric("Duplicate rows removed", f"{cl['duplicates_before']:,}")
    c[3].metric("Temperature unit fixes (F to C)", f"{cl['temperature_unit_fixes']:,}")
    st.dataframe(res["quality_table"])
    invalid = pd.Series(cl["invalid_values_by_column"], name="Invalid values removed").to_frame()
    st.write("Invalid sensor values masked, by column:")
    st.dataframe(invalid[invalid.iloc[:, 0] > 0] if (invalid.iloc[:, 0] > 0).any() else invalid)


def render_eda(res: dict) -> None:
    st.subheader("Failure rate by machine")
    fbm = res["failure_by_machine"].copy()
    fbm["failure_rate"] = fbm["failure_rate"] * 100
    st.bar_chart(fbm.set_index(config.MACHINE_ID_COL)["failure_rate"])
    st.caption("Share of readings labelled as failing within 24 hours (%).")

    left, right = st.columns(2)
    with left:
        st.subheader("Sensors vs failure")
        st.dataframe(res["sensor_summary"])
    with right:
        st.subheader("Correlation with failure")
        corr = res["sensor_correlation"].sort_values("Correlation")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(corr["Sensor"], corr["Correlation"],
                color=["#D1431F" if v > 0 else "#2F5D7C" for v in corr["Correlation"]])
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Correlation")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Failure trend")
    trend = res["failure_trend"].set_index(config.TIMESTAMP_COL)["failure_rate"] * 100
    st.line_chart(trend)

    st.subheader("Sensor trend for one machine")
    clean = res["clean"]
    machine = st.selectbox("Machine", sorted(clean[config.MACHINE_ID_COL].unique()))
    sensor = st.selectbox("Sensor", config.SENSOR_COLS, index=config.SENSOR_COLS.index(config.VIBRATION_COL))
    one = clean[clean[config.MACHINE_ID_COL] == machine].set_index(config.TIMESTAMP_COL)[sensor]
    st.line_chart(one)


def render_model(res: dict) -> None:
    m = res["primary_metrics"]
    st.subheader(f"{config.PRIMARY_MODEL_NAME}: results on unseen (test) data")
    c = st.columns(5)
    for col, key in zip(c, ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]):
        col.metric(key.replace("_", "-"), f"{m[key]:.3f}")
    st.caption("The data is split by time: the last 25% of the timeline is the test set. "
               "Scores on synthetic data are unrealistically high.")

    st.subheader("Alert threshold")
    threshold = st.slider("Probability above which a reading raises an alert", 0.05, 0.95,
                          float(config.DECISION_THRESHOLD), 0.05)
    tm = evaluation.compute_metrics(res["y_test"], res["test_prob"], threshold)
    c = st.columns(4)
    c[0].metric("Precision", f"{tm['Precision']:.3f}")
    c[1].metric("Recall", f"{tm['Recall']:.3f}")
    c[2].metric("False alarms", tm["FP"])
    c[3].metric("Missed failures", tm["FN"])
    st.line_chart(res["thresholds"].set_index("Threshold")[["Precision", "Recall"]])

    left, right = st.columns(2)
    with left:
        st.subheader("Top features")
        top = res["importance"].head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(top["Feature"], top["Importance"], color="#2A8A74")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Model comparison")
        st.dataframe(res["comparison"][["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].round(3))
        if len(res["comparison"]) == 1:
            st.info("Tick 'Compare all four models' in the sidebar to train the other models too.")


def render_risk(res: dict) -> None:
    ms = res["machine_summary"]
    c = st.columns(3)
    for col, status in zip(c, ["Healthy", "Warning", "High Risk"]):
        col.metric(status, int((ms["Risk_Status"] == status).sum()))
    st.subheader("Predicted failure risk by machine")
    ordered = ms.sort_values("Failure_Probability")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(ordered))))
    ax.barh(ordered[config.MACHINE_ID_COL], ordered["Failure_Probability"] * 100,
            color=[RISK_COLOURS[s] for s in ordered["Risk_Status"]])
    ax.set_xlabel("Failure probability at latest reading (%)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Maintenance priority list")
    table = ms[["Priority_Rank", config.MACHINE_ID_COL, "Failure_Probability", "Risk_Status", "Recommendation"]].copy()
    table["Failure_Probability"] = (table["Failure_Probability"] * 100).round(1).astype(str) + "%"
    st.dataframe(table)
    st.caption(f"Risk bands: Warning from {config.RISK_THRESHOLDS['Warning']:.0%}, "
               f"High Risk from {config.RISK_THRESHOLDS['High Risk']:.0%}.")


def render_downloads(res: dict) -> None:
    st.subheader("Download results (CSV)")
    items = {
        "machine_risk_summary.csv": res["machine_summary"],
        "scored_readings.csv": res["scored"],
        "model_comparison.csv": res["comparison"],
        "kpi_summary.csv": res["kpis"],
    }
    for name, frame in items.items():
        st.download_button(f"Download {name}", app_service.to_csv_bytes(frame), file_name=name,
                           mime="text/csv", key=name)
    st.caption("These files can be loaded into Power BI as the dashboard data sources.")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Predictive Maintenance", page_icon=str(LOGO) if LOGO.exists() else None,
                       layout="wide")
    head = st.columns([1, 8])
    if LOGO.exists():
        head[0].image(str(LOGO), width=90)
    head[1].title("Predictive Maintenance of Industrial Equipment")
    head[1].caption("Upload IoT sensor data, then explore data quality, patterns, model results "
                    "and which machines need maintenance first.")

    with st.sidebar:
        st.header("1. Provide data")
        source = st.radio("Data source", ["Upload a CSV file", "Use the sample dataset"])
        uploaded = None
        if source == "Upload a CSV file":
            uploaded = st.file_uploader("Sensor data (CSV)", type=["csv"])
            st.download_button("Download a template CSV", template_bytes(),
                               file_name="sensor_data_template.csv", mime="text/csv")
            st.caption("Required columns: " + ", ".join(config.REQUIRED_COLS))
        st.header("2. Options")
        compare_all = st.checkbox("Compare all four models (slower)", value=False)
        run = st.button("Run analysis", type="primary")

    if run:
        if source == "Upload a CSV file":
            if uploaded is None:
                st.warning("Please choose a CSV file first.")
                return
            file_bytes = uploaded.getvalue()
        else:
            file_bytes = sample_bytes()
        with st.spinner("Cleaning data, engineering features and training the model..."):
            results, error = cached_analysis(file_bytes, compare_all)
        if error:
            st.error(error)
            return
        st.session_state["results"] = results

    results = st.session_state.get("results")
    if results is None:
        st.info("Choose a data source in the sidebar and press **Run analysis**.")
        return

    tabs = st.tabs(["Overview", "Data quality", "Exploration", "Model", "Machine risk", "Downloads"])
    for tab, renderer in zip(tabs, [render_overview, render_quality, render_eda,
                                    render_model, render_risk, render_downloads]):
        with tab:
            renderer(results)


if __name__ == "__main__":
    main()
