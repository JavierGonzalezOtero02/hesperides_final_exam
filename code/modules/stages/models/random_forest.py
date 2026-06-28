"""RandomForestModel — sklearn random forest classifier.

Bagged trees: less prone to overfitting noise than boosting because trees are
grown independently rather than chasing residuals. Tree-based and therefore
scale-invariant — the upstream RobustScaler does not affect splits.

Defaults are shallow/regularized on purpose (small, noisy ~1500-row train set).
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, n_estimators: int = 400, max_depth: int = 5,
                 min_samples_leaf: int = 30, max_features="sqrt",
                 random_state: int = 42, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.model = None

    def initialize(self):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )

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
