"""SVMRBFModel — Support Vector Machine with an RBF kernel.

Scale-sensitive by design: the RBF kernel measures Euclidean distances between
samples, so features on different scales (e.g. rsi_14 in [0,100] vs returns
~0.01) would otherwise be dominated by the largest-magnitude feature. This model
genuinely exploits the upstream RobustScaler in the DPP slot.

Well suited to small datasets (the SVM is defined by a handful of support
vectors), and the RBF kernel captures non-linear feature interactions that the
linear logistic baseline cannot.
"""

import pandas as pd
from sklearn.svm import SVC


class SVMRBFModel:
    def __init__(self, C: float = 1.0, gamma="scale", **kwargs):
        self.C = C
        self.gamma = gamma
        self.model = None

    def initialize(self):
        self.model = SVC(C=self.C, kernel="rbf", gamma=self.gamma)

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
