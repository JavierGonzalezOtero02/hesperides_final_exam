"""TimeSeriesXYSplit — Constructs features and target from OHLCV data."""

import pandas as pd
import numpy as np


class TimeSeriesXYSplit:
    def __init__(self, target: str = "binary_direction_1d", sma_window: int = 20, **kwargs):
        self.target = target
        self.sma_window = sma_window

    def fit(self, X=None, y=None, metadata=None, **kwargs):
        return X, y, metadata or {}

    def transform(self, X, y=None, metadata=None, **kwargs):
        metadata = metadata or {}
        df = X.copy()
        df = df.sort_values("Date").reset_index(drop=True)

        # Compute continuous return
        df["pct_return_1d"] = df["Close"].pct_change()

        # Compute target
        if self.target == "binary_direction_1d":
            df["target"] = (df["pct_return_1d"].shift(-1) > 0).astype(int)
        elif self.target == "pct_return_1d":
            df["target"] = df["pct_return_1d"].shift(-1)
        elif self.target == "log_return_1d":
            df["target"] = np.log(df["Close"].shift(-1) / df["Close"])
        else:
            raise ValueError(f"Unknown target: {self.target}")

        # Compute SMA feature (over continuous return)
        df[f"sma_{self.sma_window}"] = df["pct_return_1d"].rolling(window=self.sma_window).mean()

        # Drop NaN rows (from SMA window + shift)
        df = df.dropna(subset=[f"sma_{self.sma_window}", "target"]).reset_index(drop=True)

        # Store close prices in metadata (aligned)
        metadata["close_prices"] = df["Close"].values
        metadata["dates"] = df["Date"].values

        # Build X and y
        y_out = df[["target"]].copy()
        X_out = df[[f"sma_{self.sma_window}"]].copy()

        # Keep Date for temporal splitting
        X_out["Date"] = df["Date"].values

        return X_out, y_out, metadata

