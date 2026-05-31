"""TemporalValidationSplit — Fixed date-based temporal split."""

import pandas as pd
import numpy as np


class TemporalValidationSplit:
    def __init__(self, val_start_date: str = "2020-01-01",
                 test_start_date: str = "2022-01-03", **kwargs):
        self.val_start_date = pd.Timestamp(val_start_date)
        self.test_start_date = pd.Timestamp(test_start_date)

    def transform(self, X, y, metadata=None, **kwargs):
        metadata = metadata or {}
        dates = pd.to_datetime(X["Date"])

        train_mask = dates < self.val_start_date
        val_mask = (dates >= self.val_start_date) & (dates < self.test_start_date)
        test_mask = dates >= self.test_start_date

        # Drop Date column — no longer needed as feature
        X_features = X.drop(columns=["Date"])

        X_train = X_features[train_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        X_val = X_features[val_mask].reset_index(drop=True)
        y_val = y[val_mask].reset_index(drop=True)
        X_test = X_features[test_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)

        # Store close prices for test period (needed by POS for strategy returns)
        if "close_prices" in metadata:
            close_prices = metadata["close_prices"]
            metadata["close_prices_test"] = close_prices[test_mask.values]

        print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test, metadata

