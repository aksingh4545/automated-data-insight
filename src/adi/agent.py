def rank_insights(candidates, top_n: int): 
    scored = []
    type_weight = {"anomaly": 1.3, "trend": 1.2, "cluster": 1.0}

    for item in candidates:
        severity = float(item.get("severity", 0.0))
        weight = type_weight.get(item.get("type"), 1.0)
        score = severity * 100.0 * weight

        if score >= 80:
            priority = "high"
        elif score >= 40:
            priority = "medium"
        else:
            priority = "low"

        action = {
            "anomaly": "Review recent anomalous transactions.",
            "trend": "Validate drivers behind the trend shift.",
            "cluster": "Explore segment strategy adjustments.",
        }.get(item.get("type"), "Review this insight.")

        enriched = dict(item)
        enriched.update({"score": score, "priority": priority, "action": action})
        scored.append(enriched)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
