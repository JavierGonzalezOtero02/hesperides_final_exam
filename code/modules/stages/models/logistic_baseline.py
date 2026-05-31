"""LogisticBaselineModel — Baseline binary classifier for market direction."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticBaselineModel:
    def __init__(self, C: float = 1.0, max_iter: int = 1000, **kwargs):
        self.C = C
        self.max_iter = max_iter
        self.model = None

    def initialize(self):
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            X_test=None, y_test=None, metadata=None, **kwargs):
        metadata = metadata or {}
        self.model.fit(X_train.values, y_train.values.ravel())
        return None, metadata

    def predict(self, X, y=None, metadata=None, **kwargs):
        metadata = metadata or {}
        preds = self.model.predict(X.values)
        y_pred = pd.DataFrame({"target": preds})
        return X, y_pred, metadata

