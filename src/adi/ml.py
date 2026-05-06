import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def prepare_features(df: pd.DataFrame, feature_cols=None):
    if feature_cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = [c for c in feature_cols if c in df.columns]

    if not cols:
        raise ValueError("No numeric features available.")

    x = df[cols].copy().fillna(0.0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, cols


def run_kmeans(x_scaled, n_clusters: int, random_state: int = 42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(x_scaled)
    return labels, model


def run_isolation_forest(x_scaled, contamination: float, random_state: int = 42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(x_scaled)
    scores = model.decision_function(x_scaled)
    return preds, scores, model


def compute_trends(
    df: pd.DataFrame,
    window_days: int,
    total_col: str,
    id_col: str | None = None,
    timestamp_col: str = "timestamp",
):
    if timestamp_col not in df.columns:
        raise ValueError("timestamp column required.")

    if total_col not in df.columns:
        raise ValueError("total metric column required.")

    count_col = id_col if id_col in df.columns else None

    daily = (
        df.set_index(timestamp_col)
        .resample("D")
        .agg(
            total_sum=(total_col, "sum"),
            txn_count=(count_col or total_col, "count"),
        )
        .fillna(0.0)
    )

    if len(daily) == 0:
        return {
            "recent_total": 0.0,
            "prior_total": 0.0,
            "pct_change": 0.0,
            "recent_count": 0,
            "prior_count": 0,
            "count_change": 0.0,
        }

    recent = daily.tail(window_days)
    prior = daily.iloc[:-window_days].tail(window_days)

    recent_total = float(recent["total_sum"].sum())
    prior_total = float(prior["total_sum"].sum())
    recent_count = int(recent["txn_count"].sum())
    prior_count = int(prior["txn_count"].sum())

    pct_change = (recent_total - prior_total) / prior_total if prior_total > 0 else 0.0
    count_change = (recent_count - prior_count) / prior_count if prior_count > 0 else 0.0

    return {
        "recent_total": recent_total,
        "prior_total": prior_total,
        "pct_change": pct_change,
        "recent_count": recent_count,
        "prior_count": prior_count,
        "count_change": count_change,
    }
