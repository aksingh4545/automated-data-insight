from adi.agent import rank_insights


def test_rank_insights_prioritizes_high_score():
    candidates = [
        {"type": "cluster", "severity": 0.1, "message": "low"},
        {"type": "anomaly", "severity": 0.9, "message": "high"},
    ]

    top = rank_insights(candidates, top_n=1)
    assert top[0]["type"] == "anomaly"
    assert top[0]["priority"] in {"high", "medium"}
