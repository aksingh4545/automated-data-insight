import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from .config import DATA_PATH, settings
    from .data_loader import add_timestamp, load_data
    from .pipeline import run_pipeline
except ImportError: 
    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from adi.config import DATA_PATH, settings
    from adi.data_loader import add_timestamp, load_data
    from adi.pipeline import run_pipeline

st.set_page_config(page_title="Autonomous Data Insight", layout="wide")

st.title("Autonomous Data Insight")

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    n_clusters = st.slider("Clusters", 2, 8, settings.n_clusters)
    contamination = st.slider(
        "Anomaly Contamination", 0.01, 0.2, settings.anomaly_contamination
    )
    trend_window = st.slider("Trend Window (days)", 3, 30, settings.trend_window_days)

    st.subheader("Columns")
    date_col = st.text_input("Date column (optional)", value="date")
    time_col = st.text_input("Time column (optional)", value="time")
    timestamp_col = st.text_input("Timestamp column name", value="timestamp")
    id_col = st.text_input("ID column (optional)", value="invoice_id")

    st.subheader("GenAI")
    use_ollama = st.checkbox("Use Ollama", value=True)
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    ollama_model = st.text_input("Ollama Model", value="phi3")

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_data(DATA_PATH)

if date_col not in df.columns:
    date_col = None
if time_col not in df.columns:
    time_col = None

if timestamp_col in df.columns and date_col is None and time_col is None:
    df = df.copy()
else:
    df = add_timestamp(df, date_col=date_col, time_col=time_col)
    if "timestamp" in df.columns and timestamp_col != "timestamp":
        df = df.rename(columns={"timestamp": timestamp_col})
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found. Please upload a dataset with numeric fields.")
    st.stop()

with st.sidebar:
    feature_cols = st.multiselect(
        "Feature columns for ML",
        options=numeric_cols,
        default=numeric_cols,
    )
    total_col = st.selectbox(
        "Total/metric column",
        options=numeric_cols,
        index=numeric_cols.index("total") if "total" in numeric_cols else 0,
    )
    scatter_x = st.selectbox(
        "Scatter X", options=numeric_cols, index=numeric_cols.index(total_col)
    )
    scatter_y = st.selectbox(
        "Scatter Y", options=numeric_cols, index=0 if numeric_cols else 0
    )

result = run_pipeline(
    df=df,
    overrides={
        "n_clusters": n_clusters,
        "anomaly_contamination": contamination,
        "trend_window_days": trend_window,
    },
    feature_cols=feature_cols,
    total_col=total_col,
    id_col=id_col if id_col in df.columns else None,
    timestamp_col=timestamp_col,
    genai_options={
        "use_ollama": use_ollama,
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
    },
)

st.subheader("Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Anomalies", f"{int((result['anomaly_preds'] == -1).sum()):,}")
col3.metric("Clusters", f"{n_clusters}")

st.subheader("Visualizations")

plot_df = df.copy()
plot_df["cluster"] = result["labels"]
plot_df["anomaly"] = result["anomaly_preds"] == -1

cluster_fig = px.scatter(
    plot_df,
    x=scatter_x,
    y=scatter_y,
    color="cluster",
    hover_data=[c for c in ["product_line", "city"] if c in plot_df.columns],
    title="Cluster Scatter",
)

anomaly_fig = px.scatter(
    plot_df,
    x=scatter_x,
    y=scatter_y,
    color="anomaly",
    hover_data=[c for c in ["product_line", "city"] if c in plot_df.columns],
    title="Anomaly Scatter",
)

trend = result["trends"]

st.plotly_chart(cluster_fig, use_container_width=True)
st.plotly_chart(anomaly_fig, use_container_width=True)

if trend:
    trend_fig = px.bar(
        x=["Prior Window", "Recent Window"],
        y=[trend["prior_total"], trend["recent_total"]],
        title="Trend Comparison",
    )
    st.plotly_chart(trend_fig, use_container_width=True)
else:
    st.info("Trend chart unavailable (timestamp or metric column missing).")

st.subheader("Top Insights")
for item in result["top_insights"]:
    st.markdown(
        f"**{item['priority'].upper()}** - {item['message']}  \\nScore: {item['score']:.1f} | Action: {item['action']}"
    )

st.subheader("AI Explanation")
st.write(result["summary"])
