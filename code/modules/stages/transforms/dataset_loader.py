"""DatasetLoader — Loads dataset CSV from disk."""

from pathlib import Path
import pandas as pd


class DatasetLoader:
    def __init__(self, data_path: str = "data", dataset_name: str = "dataset", **kwargs):
        self.path = Path(data_path) / dataset_name / "table" / "table.csv"

    def fit(self, X=None, y=None, metadata=None, **kwargs):
        return X, y, metadata or {}

    def transform(self, X=None, y=None, metadata=None, **kwargs):
        metadata = metadata or {}
        df = pd.read_csv(self.path, parse_dates=["Date"])
        return df, None, metadata

