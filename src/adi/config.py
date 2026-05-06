from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "sales.csv"


@dataclass(frozen=True)
class Settings:
    n_clusters: int = 4
    anomaly_contamination: float = 0.05
    trend_window_days: int = 7
    trend_spike_pct: float = 0.25
    anomaly_recent_days: int = 7
    top_insights: int = 5


settings = Settings()
