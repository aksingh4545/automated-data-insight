import pandas as pd

from .config import Settings


def generate_insight_candidates(
    df: pd.DataFrame,
    labels,
    anomaly_preds,
    anomaly_scores,
    trends: dict,
    settings: Settings,
    total_col: str,
    timestamp_col: str = "timestamp",
):
    candidates = []
    working = df.copy()
    working["cluster"] = labels
    working["is_anomaly"] = anomaly_preds == -1
    working["anomaly_score"] = anomaly_scores

    if total_col not in working.columns:
        return candidates

    overall_mean = working[total_col].mean()
    cluster_means = working.groupby("cluster")[total_col].mean()

    for cluster_id, mean_total in cluster_means.items():
        if overall_mean == 0:
            continue
        ratio = mean_total / overall_mean
        if ratio >= 1.5 or ratio <= 0.7:
            candidates.append(
                {
                    "type": "cluster",
                    "cluster": int(cluster_id),
                    "metric": "avg_total",
                    "value": float(mean_total),
                    "comparison": float(ratio),
                    "severity": float(abs(ratio - 1.0)),
                    "message": f"Cluster {cluster_id} has {ratio:.2f}x average spend vs overall.",
                }
            )

    if timestamp_col in working.columns:
        recent_cutoff = working[timestamp_col].max() - pd.Timedelta(
            days=settings.anomaly_recent_days
        )
        recent_mask = (working[timestamp_col] >= recent_cutoff) & (
            working["is_anomaly"]
        )
        recent_anoms = working[recent_mask]
        recent_total = working[working[timestamp_col] >= recent_cutoff]
        anomaly_rate = (
            len(recent_anoms) / len(recent_total) if len(recent_total) else 0.0
        )

        if len(recent_anoms) >= 3:
            candidates.append(
                {
                    "type": "anomaly",
                    "metric": "recent_anomaly_count",
                    "value": int(len(recent_anoms)),
                    "comparison": float(anomaly_rate),
                    "severity": float(anomaly_rate),
                    "message": f"{len(recent_anoms)} anomalies detected in the last {settings.anomaly_recent_days} days.",
                }
            )

    pct_change = trends.get("pct_change")
    if pct_change is not None and abs(pct_change) >= settings.trend_spike_pct:
        direction = "increase" if pct_change > 0 else "decrease"
        candidates.append(
            {
                "type": "trend",
                "metric": "total_sales_change",
                "value": float(pct_change),
                "comparison": float(pct_change),
                "severity": float(abs(pct_change)),
                "message": f"Sales show a {direction} of {pct_change:.1%} vs previous period.",
            }
        )

    return candidates
