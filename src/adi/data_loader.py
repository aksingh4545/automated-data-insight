import pandas as pd

from .config import DATA_PATH


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def add_timestamp(df: pd.DataFrame, date_col=None, time_col=None):
    working = df.copy()
    if date_col and time_col:
        ts = pd.to_datetime(
            working[date_col].astype(str) + " " + working[time_col].astype(str),
            errors="coerce",
        )
        working["timestamp"] = ts
    elif date_col:
        working["timestamp"] = pd.to_datetime(working[date_col], errors="coerce")

    if "timestamp" in working.columns:
        working = working.dropna(subset=["timestamp"])

    return working
