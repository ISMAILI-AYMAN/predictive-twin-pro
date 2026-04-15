from __future__ import annotations

import os
from datetime import datetime
from typing import Dict

import altair as alt
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

INFERENCE_API = os.getenv("INFERENCE_API_URL", "http://localhost:8000")


def fetch_metrics() -> Dict[str, float]:
    response = requests.get(f"{INFERENCE_API}/metrics", timeout=3)
    response.raise_for_status()
    return response.json()


def fetch_history(limit: int = 200) -> Dict[str, object]:
    response = requests.get(f"{INFERENCE_API}/metrics/history", params={"limit": limit}, timeout=3)
    response.raise_for_status()
    return response.json()


def apply_time_window(df: pd.DataFrame, window_key: str) -> pd.DataFrame:
    if df.empty or window_key == "all":
        return df
    now = pd.Timestamp.utcnow()
    if window_key == "15m":
        start = now - pd.Timedelta(minutes=15)
    elif window_key == "1h":
        start = now - pd.Timedelta(hours=1)
    else:
        start = now - pd.Timedelta(hours=24)
    return df[df["event_dt"] >= start]


def compute_events_per_minute(df: pd.DataFrame) -> float:
    if len(df) <= 1:
        return float(len(df))
    span_minutes = max((df["event_dt"].max() - df["event_dt"].min()).total_seconds() / 60.0, 1.0)
    return len(df) / span_minutes


def severity_badge(label: str, value: int, color: str) -> str:
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{color};color:white;font-size:0.85rem;margin-right:8px;'>"
        f"{label}: {value}</span>"
    )


st.set_page_config(page_title="Predictive Twin Dashboard", layout="wide")
st.title("Predictive Twin Pro Dashboard")
st.caption("System health, model health, and asset health in one place.")

controls_col1, controls_col2, controls_col3 = st.columns([1.2, 1.2, 2.6])
auto_refresh_enabled = controls_col1.toggle("Auto-refresh", value=False)
auto_refresh_interval = controls_col2.selectbox(
    "Interval (sec)",
    options=[5, 10, 15, 30, 60],
    index=1,
)
if controls_col3.button("Refresh metrics now"):
    st.rerun()

if auto_refresh_enabled:
    st_autorefresh(interval=int(auto_refresh_interval) * 1000, key="dashboard_auto_refresh")

try:
    data = fetch_metrics()
    history_payload = fetch_history(limit=1000)
    history_events = history_payload.get("events", [])

    history_df = pd.DataFrame(history_events)
    if not history_df.empty:
        local_tz = datetime.now().astimezone().tzinfo
        history_df["event_dt"] = pd.to_datetime(history_df["event_time"], unit="s", utc=True).dt.tz_convert(
            local_tz
        )
        history_df = history_df.sort_values("event_dt")
    else:
        history_df = pd.DataFrame(
            columns=[
                "id",
                "event_time",
                "asset_id",
                "anomaly_score",
                "rul_hours",
                "drift_severity",
                "drift_score",
                "model_backend",
                "event_dt",
            ]
        )

    st.subheader("Filters")
    f_col1, f_col2, f_col3 = st.columns(3)
    window_choice = f_col1.selectbox(
        "Time Window",
        options=["15m", "1h", "24h", "all"],
        index=1,
    )
    assets = sorted(history_df["asset_id"].dropna().unique().tolist()) if not history_df.empty else []
    asset_choice = f_col2.selectbox(
        "Asset",
        options=["all", *assets],
        index=0,
    )
    severity_choice = f_col3.selectbox(
        "Drift Severity",
        options=["all", "info", "warn", "critical"],
        index=0,
    )

    filtered_df = apply_time_window(history_df, window_choice)
    if asset_choice != "all":
        filtered_df = filtered_df[filtered_df["asset_id"] == asset_choice]
    if severity_choice != "all":
        filtered_df = filtered_df[filtered_df["drift_severity"] == severity_choice]

    latest_event = filtered_df.iloc[-1].to_dict() if not filtered_df.empty else None
    model_backend = (
        str(latest_event["model_backend"])
        if latest_event is not None
        else ("dl_trained" if float(data.get("model_backend_dl", 0.0)) == 1.0 else "fallback")
    )
    prediction_rate = compute_events_per_minute(filtered_df)
    info_count = int((filtered_df["drift_severity"] == "info").sum()) if not filtered_df.empty else 0
    warn_count = int((filtered_df["drift_severity"] == "warn").sum()) if not filtered_df.empty else 0
    critical_count = int((filtered_df["drift_severity"] == "critical").sum()) if not filtered_df.empty else 0

    st.subheader("Prediction Health")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Latest Anomaly Score", f"{data.get('anomaly_score_latest', 0):.4f}")
    p_col2.metric("Latest RUL (h)", f"{data.get('rul_latest', 0):.2f}")
    p_col3.metric("Model Backend", model_backend)
    p_col4.metric("Prediction Rate (events/min)", f"{prediction_rate:.2f}")
    st.markdown(
        severity_badge("info", info_count, "#10B981")
        + severity_badge("warn", warn_count, "#F59E0B")
        + severity_badge("critical", critical_count, "#EF4444"),
        unsafe_allow_html=True,
    )

    a_col1, a_col2, a_col3 = st.columns(3)
    a_col1.metric("Requests", f"{int(data.get('requests_total', 0))}")
    a_col2.metric("Drift Warn Total", f"{int(data.get('drift_warn_total', 0))}")
    a_col3.metric("Drift Critical Total", f"{int(data.get('drift_critical_total', 0))}")

    st.subheader("Persisted History")
    st.caption(f"Stored events: {history_payload.get('count', 0)}")

    if not filtered_df.empty:
        anomaly_base = (
            alt.Chart(filtered_df)
            .mark_line(color="#9CA3AF")
            .encode(
                x=alt.X("event_dt:T", title="Time"),
                y=alt.Y("anomaly_score:Q", title="Anomaly Score"),
            )
        )
        anomaly_points = (
            alt.Chart(filtered_df)
            .mark_circle(size=55)
            .encode(
                x=alt.X("event_dt:T", title="Time"),
                y=alt.Y("anomaly_score:Q", title="Anomaly Score"),
                color=alt.Color(
                    "drift_severity:N",
                    scale=alt.Scale(
                        domain=["info", "warn", "critical"],
                        range=["#10B981", "#F59E0B", "#EF4444"],
                    ),
                ),
                tooltip=[
                    "asset_id:N",
                    "drift_severity:N",
                    "anomaly_score:Q",
                    "rul_hours:Q",
                    alt.Tooltip("event_dt:T", title="Local Time", format="%Y-%m-%d %H:%M:%S"),
                ],
            )
        )
        st.altair_chart(anomaly_base + anomaly_points, use_container_width=True)

        filtered_df = filtered_df.copy()
        filtered_df["rul_moving_avg_10"] = filtered_df["rul_hours"].rolling(window=10, min_periods=1).mean()
        rul_chart = (
            alt.Chart(filtered_df)
            .transform_fold(["rul_hours", "rul_moving_avg_10"], as_=["series", "value"])
            .mark_line()
            .encode(
                x=alt.X("event_dt:T", title="Time"),
                y=alt.Y("value:Q", title="RUL (hours)"),
                color=alt.Color("series:N", title="Series"),
                tooltip=[alt.Tooltip("event_dt:T", title="Local Time", format="%Y-%m-%d %H:%M:%S"), "series:N", "value:Q"],
            )
        )
        st.altair_chart(rul_chart, use_container_width=True)

        timeline_df = filtered_df.copy()
        timeline_df["time_bucket"] = timeline_df["event_dt"].dt.floor("min")
        severity_counts = (
            timeline_df.groupby(["time_bucket", "drift_severity"]).size().reset_index(name="count")
        )
        drift_timeline = (
            alt.Chart(severity_counts)
            .mark_bar()
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("count:Q", title="Events"),
                color=alt.Color(
                    "drift_severity:N",
                    scale=alt.Scale(
                        domain=["info", "warn", "critical"],
                        range=["#10B981", "#F59E0B", "#EF4444"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("time_bucket:T", title="Local Minute", format="%Y-%m-%d %H:%M"),
                    "drift_severity:N",
                    "count:Q",
                ],
            )
        )
        st.subheader("Drift Severity Timeline")
        st.altair_chart(drift_timeline, use_container_width=True)

        display_df = filtered_df.sort_values("event_dt", ascending=False).copy()
        display_df["event_time"] = display_df["event_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display_df = display_df[
            [
                "id",
                "event_time",
                "asset_id",
                "anomaly_score",
                "rul_hours",
                "drift_severity",
                "drift_score",
                "model_backend",
            ]
        ]
        st.subheader("Investigation Table")
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export filtered history as CSV",
            data=csv_bytes,
            file_name="metrics_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No events match the current filters. Adjust filters or send /infer requests.")
except Exception as exc:
    st.error(f"Could not fetch metrics from inference API: {exc}")
