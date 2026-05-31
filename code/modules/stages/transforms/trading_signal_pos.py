"""TradingSignalPOS — Converts model predictions to trading signals. IMMUTABLE."""

import numpy as np
import pandas as pd


class TradingSignalPOS:
    def __init__(self, signal_mode: str = "classification", **kwargs):
        self.signal_mode = signal_mode

    def fit(self, X=None, y=None, metadata=None, **kwargs):
        return X, y, metadata or {}

    def transform(self, X, y_pred, metadata=None, **kwargs):
        metadata = metadata or {}

        # Convert y_pred to numpy array
        if isinstance(y_pred, pd.DataFrame):
            preds = y_pred.values.flatten()
        else:
            preds = np.array(y_pred).flatten()

        # Generate trading signal
        if self.signal_mode == "classification":
            # 1 → +1 (long), 0 → -1 (short)
            signal = 2 * preds - 1
        elif self.signal_mode == "regression":
            # sign of prediction
            signal = np.sign(preds)
        else:
            raise ValueError(f"Unknown signal_mode: {self.signal_mode}")

        # Compute strategy returns using test close prices
        close_prices_test = metadata.get("close_prices_test")
        if close_prices_test is not None:
            # Daily returns of the underlying
            real_returns = np.diff(close_prices_test) / close_prices_test[:-1]
            # Align: signal[t] applies to return from t to t+1
            # We need signal and returns to be same length
            min_len = min(len(signal), len(real_returns))
            strategy_returns = signal[:min_len] * real_returns[:min_len]
            metadata["strategy_returns"] = strategy_returns

        return X, y_pred, metadata

