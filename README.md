# Autonomous Data Insight

Autonomous Data Insight is a Python analytics application that turns tabular sales data into ranked, human-readable business insights. It combines data preparation, clustering, anomaly detection, trend analysis, and optional local GenAI summarization in a Streamlit dashboard.

## Features

- Load the included sample sales dataset or upload your own CSV file.
- Automatically build timestamps from separate date and time columns.
- Select numeric feature columns for machine-learning analysis.
- Segment records with K-Means clustering.
- Detect unusual records with Isolation Forest anomaly detection.
- Compare recent metric totals against a prior time window.
- Rank generated insights by severity and type.
- Visualize clusters, anomalies, and trend changes with Plotly.
- Optionally summarize insights with a local Ollama model, with a built-in deterministic fallback summary when Ollama is unavailable.

## Repository layout

```text
.
├── data/
│   └── sales.csv              # Sample sales dataset used by default
├── src/adi/
│   ├── agent.py               # Insight ranking and priority assignment
│   ├── app.py                 # Streamlit web application
│   ├── config.py              # Default paths and analysis settings
│   ├── data_loader.py         # CSV loading and timestamp preparation
│   ├── genai.py               # Local and Ollama-backed text summaries
│   ├── insights.py            # Insight candidate generation
│   ├── ml.py                  # Feature preparation, clustering, anomaly detection, trends
│   └── pipeline.py            # End-to-end analysis orchestration
├── tests/                     # Pytest suite
└── requirements.txt           # Python dependencies
```

## Requirements

- Python 3.10 or newer is recommended.
- Optional: [Ollama](https://ollama.com/) running locally if you want model-generated explanations.

## Getting started

1. Clone the repository and enter the project directory.

   ```bash
   git clone <repository-url>
   cd automated-data-insight
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies.

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Run the Streamlit app.

   ```bash
   streamlit run src/adi/app.py
   ```

5. Open the local Streamlit URL shown in your terminal.

## Using the dashboard

The sidebar controls how the analysis runs:

- **Upload CSV**: Use your own data instead of `data/sales.csv`.
- **Clusters**: Set the number of K-Means segments.
- **Anomaly Contamination**: Estimate the expected proportion of anomalies.
- **Trend Window (days)**: Choose the recent and prior windows used for trend comparisons.
- **Columns**: Configure date, time, timestamp, and identifier columns.
- **Feature columns for ML**: Pick numeric columns used for clustering and anomaly detection.
- **Total/metric column**: Pick the numeric metric used for trend and insight calculations.
- **GenAI**: Enable Ollama and configure its URL/model for AI-written summaries.

For best results, uploaded CSV files should contain at least one numeric column. Trend analysis also requires either an existing timestamp column or date/time columns that can be converted to a timestamp.

## Optional Ollama summaries

By default, the app exposes an Ollama toggle in the sidebar. When enabled, it sends the ranked insights to the configured local Ollama endpoint, for example:

```text
http://localhost:11434
```

If Ollama is disabled, unreachable, or returns an invalid response, the application falls back to a local rule-based summary so the dashboard remains usable.

## Running tests

Run the test suite with:

```bash
pytest
```

The tests cover insight ranking and insight candidate generation.

## Programmatic usage

You can also run the analysis pipeline from Python:

```python
from adi.pipeline import run_pipeline

result = run_pipeline()

for insight in result["top_insights"]:
    print(insight["priority"], insight["message"])

print(result["summary"])
```

When running scripts outside Streamlit, make sure `src` is on your `PYTHONPATH` or install the project in editable mode if packaging is added later:

```bash
PYTHONPATH=src python your_script.py
```

## Configuration defaults

Default analysis settings live in `src/adi/config.py`:

- `n_clusters`: `4`
- `anomaly_contamination`: `0.05`
- `trend_window_days`: `7`
- `trend_spike_pct`: `0.25`
- `anomaly_recent_days`: `7`
- `top_insights`: `5`

The Streamlit app lets you override several of these values at runtime.

## Data notes

The included sample data is a sales CSV with invoice, branch, customer, product, pricing, tax, total, date, time, payment, margin, income, and rating fields. You can replace it at runtime through the dashboard uploader without modifying repository files.
