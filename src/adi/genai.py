import json

import requests


def _local_summary(insights):
    sentences = []
    for item in insights:
        itype = item.get("type")
        if itype == "anomaly":
            text = (
                f"A spike in unusual transactions was detected. "
                f"{item.get('value')} anomalies occurred recently, which may require review."
            )
        elif itype == "trend":
            text = (
                f"Transaction totals show a {item.get('value'):.1%} change compared to the prior period. "
                f"This shift may reflect demand or operational changes."
            )
        elif itype == "cluster":
            text = (
                f"Customer segment {item.get('cluster')} differs notably in average spending, "
                f"which suggests a distinct behavior pattern."
            )
        else:
            text = item.get("message", "An insight was detected.")

        sentences.append(text)

    summary = " ".join(sentences) if sentences else "No significant insights were detected."
    return sentences, summary


def _ollama_summary(insights, model: str, base_url: str, timeout: int = 20):
    prompt = (
        "You are a business analyst. Summarize the following insights into a concise, "
        "human-readable paragraph with suggested implications.\n\n"
        f"Insights JSON:\n{json.dumps(insights, indent=2)}\n"
    )

    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    text = data.get("response", "").strip()
    if not text:
        raise ValueError("Empty response from Ollama.")
    return [text], text


def generate_insight_texts(
    insights,
    use_ollama: bool = False,
    ollama_model: str = "phi3",
    ollama_url: str = "http://localhost:11434",
):
    if use_ollama:
        try:
            return _ollama_summary(insights, ollama_model, ollama_url)
        except Exception:
            return _local_summary(insights)

    return _local_summary(insights)
