import pandas as pd

from adi.config import Settings
from adi.insights import generate_insight_candidates


def test_generate_insight_candidates_cluster():
    df = pd.DataFrame(
        {
            "total": [100, 110, 120, 10, 12, 11],
            "timestamp": pd.date_range("2023-01-01", periods=6, freq="D"),
        }
    )
    labels = [0, 0, 0, 1, 1, 1]
    anomaly_preds = [1, 1, 1, 1, 1, 1]
    anomaly_scores = [0.1] * 6
    trends = {"pct_change": 0.0}

    candidates = generate_insight_candidates(
        df,
        labels,
        anomaly_preds,
        anomaly_scores,
        trends,
        Settings(),
        total_col="total",
        timestamp_col="timestamp",
    )

    assert any(c["type"] == "cluster" for c in candidates)


def test_generate_insight_candidates_trend():
    df = pd.DataFrame(
        {
            "total": [100, 110, 120, 130],
            "timestamp": pd.date_range("2023-01-01", periods=4, freq="D"),
        }
    )
    labels = [0, 0, 0, 0]
    anomaly_preds = [1, 1, 1, 1]
    anomaly_scores = [0.1] * 4
    trends = {"pct_change": 0.5}

    candidates = generate_insight_candidates(
        df,
        labels,
        anomaly_preds,
        anomaly_scores,
        trends,
        Settings(),
        total_col="total",
        timestamp_col="timestamp",
    )

    assert any(c["type"] == "trend" for c in candidates)
