"""HistGradientBoostingModel — sklearn histogram-based gradient boosting.

A LightGBM-style learner with zero extra dependencies (ships with scikit-learn).
Tree-based and therefore scale-invariant: the upstream RobustScaler in the DPP
slot is harmless but does not affect splits.

Defaults are regularized on purpose: the train set is small (~1500 rows) and the
weekly-direction target is very noisy, so an unconstrained booster overfits fast.
Early stopping uses an internal validation split carved from the training data.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class HistGradientBoostingModel:
    def __init__(self, learning_rate: float = 0.05, max_leaf_nodes: int = 15,
                 min_samples_leaf: int = 50, l2_regularization: float = 1.0,
                 max_iter: int = 300, n_iter_no_change: int = 20,
                 validation_fraction: float = 0.15, random_state: int = 42,
                 **kwargs):
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_iter = max_iter
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.model = None

    def initialize(self):
        self.model = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
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
