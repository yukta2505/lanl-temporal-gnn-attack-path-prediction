"""Streamlit dashboard for model next-hop predictions and asset risk ranking."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.inference.risk_scoring import score_risk  # noqa: E402


st.set_page_config(page_title="Attack Path Prioritization", layout="wide")
st.title("Explainable Lateral Movement Alerts")
st.caption("Next-hop predictions ranked by model probability and supplied asset criticality.")

predictions_file = st.file_uploader("Predictions CSV", type="csv", key="predictions")
assets_file = st.file_uploader("Asset criticality CSV", type="csv", key="assets")
st.markdown("Predictions need `source_host`, `target_host`, `probability`; optional: `explanation`. "
            "Asset inventory needs `host`, `criticality` on a 0–5 scale.")

if predictions_file and assets_file:
    predictions = pd.read_csv(predictions_file)
    assets = pd.read_csv(assets_file)
    required_predictions = {"source_host", "target_host", "probability"}
    missing = required_predictions - set(predictions.columns)
    if missing:
        st.error("Predictions CSV is missing: " + ", ".join(sorted(missing)))
        st.stop()
    if not {"host", "criticality"}.issubset(assets.columns):
        st.error("Asset CSV must contain `host` and `criticality` columns.")
        st.stop()
    if assets.host.duplicated().any():
        st.error("Asset CSV contains duplicate host names; provide one criticality per host.")
        st.stop()

    source_options = sorted(predictions["source_host"].dropna().astype(str).unique())
    selected_source = st.selectbox("Compromised/source host", ["All sources", *source_options])
    top_k = st.number_input("Show top alerts", min_value=1, max_value=1000, value=50, step=10)
    if selected_source != "All sources":
        predictions = predictions[predictions["source_host"].astype(str) == selected_source]

    assets["criticality"] = pd.to_numeric(assets["criticality"], errors="coerce")
    ranked = predictions.merge(assets[["host", "criticality"]],
                              left_on="target_host", right_on="host", how="left")
    invalid_probability = ~pd.to_numeric(ranked.probability, errors="coerce").between(0, 1)
    invalid_criticality = ~ranked.criticality.between(0, 5)
    ranked["risk_score"] = pd.NA
    ranked["risk_level"] = "Unknown criticality"
    valid = ~invalid_probability & ~invalid_criticality
    for index in ranked.index[valid]:
        assessment = score_risk(float(ranked.at[index, "probability"]),
                                float(ranked.at[index, "criticality"]))
        ranked.at[index, "risk_score"] = round(assessment.risk_score, 2)
        ranked.at[index, "risk_level"] = assessment.risk_level

    prioritized = ranked[valid].sort_values("risk_score", ascending=False).head(int(top_k))
    unknown = ranked[~valid]
    a, b, c = st.columns(3)
    a.metric("Prioritized predictions", len(prioritized))
    b.metric("Missing/invalid criticality", len(unknown))
    c.metric("Highest risk score", f"{prioritized.risk_score.max():.2f}" if not prioritized.empty else "—")
    st.subheader("Prioritized alerts")
    columns = ["source_host", "target_host", "probability", "criticality", "risk_score", "risk_level"]
    if "explanation" in prioritized.columns:
        columns.append("explanation")
    st.dataframe(prioritized[columns], use_container_width=True, hide_index=True)
    if not unknown.empty:
        st.subheader("Predictions needing asset inventory values")
        st.dataframe(unknown.drop(columns=["host"], errors="ignore"), use_container_width=True, hide_index=True)
    st.download_button("Download ranked alerts", prioritized[columns].to_csv(index=False),
                       "ranked_alerts.csv", "text/csv")
else:
    st.info("Upload model predictions and an asset inventory to rank the alerts.")
