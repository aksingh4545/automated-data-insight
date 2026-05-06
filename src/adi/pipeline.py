from .agent import rank_insights
from .config import DATA_PATH, settings
from .data_loader import load_data
from .genai import generate_insight_texts
from .insights import generate_insight_candidates
from .ml import compute_trends, prepare_features, run_isolation_forest, run_kmeans


def run_pipeline(
    data_path=None,
    df=None,
    overrides=None,
    feature_cols=None,
    total_col=None,
    id_col=None,
    timestamp_col="timestamp",
    genai_options=None,
):
    cfg = settings
    if overrides:
        cfg = cfg.__class__(**{**cfg.__dict__, **overrides})

    if df is None:
        df = load_data(path=data_path or DATA_PATH)

    x_scaled, used_features = prepare_features(df, feature_cols=feature_cols)

    labels, _ = run_kmeans(x_scaled, cfg.n_clusters)
    anomaly_preds, anomaly_scores, _ = run_isolation_forest(
        x_scaled, cfg.anomaly_contamination
    )

    trends = {}
    if timestamp_col in df.columns and total_col:
        trends = compute_trends(
            df,
            cfg.trend_window_days,
            total_col=total_col,
            id_col=id_col,
            timestamp_col=timestamp_col,
        )

    candidates = generate_insight_candidates(
        df,
        labels,
        anomaly_preds,
        anomaly_scores,
        trends,
        cfg,
        total_col=total_col or used_features[0],
        timestamp_col=timestamp_col,
    )
    top_insights = rank_insights(candidates, cfg.top_insights)
    sentences, summary = generate_insight_texts(
        top_insights, **(genai_options or {})
    )

    return {
        "data": df,
        "labels": labels,
        "anomaly_preds": anomaly_preds,
        "anomaly_scores": anomaly_scores,
        "trends": trends,
        "candidates": candidates,
        "top_insights": top_insights,
        "sentences": sentences,
        "summary": summary,
        "used_features": used_features,
    }
