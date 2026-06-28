"""RobustScalerTransform — DPP step: median/IQR scaling of features.

Fit on train only (the pipeline enforces this), then applied to train/val/test.
RobustScaler (median + IQR) is preferred over StandardScaler for financial
returns: the train period (1990-2019) contains fat-tailed outliers (2008,
March 2020) that would distort a mean/std-based scaler.

Only feature columns are scaled; metadata (close prices, etc.) is untouched.
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler


class RobustScalerTransform:
    def __init__(self, **kwargs):
        self.scaler = RobustScaler()
        self.cols = None

    def fit(self, X, y=None, metadata=None, **kwargs):
        # Date column was already dropped by TemporalValidationSplit, so every
        # column in X is a numeric feature.
        self.cols = list(X.columns)
        self.scaler.fit(X.values)
        return X, y, metadata or {}

    def transform(self, X, y=None, metadata=None, **kwargs):
        X_scaled = pd.DataFrame(
            self.scaler.transform(X[self.cols].values),
            columns=self.cols,
            index=X.index,
        )
        return X_scaled, y, metadata or {}
