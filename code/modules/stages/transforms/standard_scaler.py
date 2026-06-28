"""StandardScalerTransform — z-scores features; fit on TRAIN only.

The baseline fed raw, unscaled features into LogisticRegression, whose L2 penalty
(controlled by ``C``) implicitly assumes standardized inputs — features on different
scales are penalized unequally. This DPP stage fixes that: it learns the mean/std on
the training split only (``fit`` is called by the orchestrator with X_train), and
applies the *same* learned statistics to val/test. The fitted scaler travels inside
the saved bundle (the ``dpp`` list), so ``predict`` reuses train statistics — no
test information ever leaks into the transform.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


class StandardScalerTransform:
    def __init__(self, **kwargs):
        self.scaler = StandardScaler()
        self._cols = None

    def fit(self, X, y=None, metadata=None, **kwargs):
        # "Date" is already dropped by TemporalValidationSplit before DPP runs, but
        # guard anyway so the stage is safe on any upstream ordering.
        self._cols = [c for c in X.columns if c != "Date"]
        self.scaler.fit(X[self._cols].values)
        return X, y, metadata or {}

    def transform(self, X, y=None, metadata=None, **kwargs):
        X = X.copy()
        # Use the exact columns learned at fit time, in the same order — guarantees
        # train and predict apply the transform identically.
        X[self._cols] = self.scaler.transform(X[self._cols].values)
        return X, y, metadata or {}
