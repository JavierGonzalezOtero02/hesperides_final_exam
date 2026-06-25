"""BalancedLogisticModel — class-balanced logistic regression for weekly direction.

Same model family as the baseline (L2 logistic regression) but with one deliberate,
well-motivated change: ``class_weight="balanced"``.

Rationale. The S&P 500 rises in ~54% of weeks (positive long-run drift), so a plain
classifier trained on 1990-2019 learns a majority-"up" prior and tends to predict
"long" almost unconditionally. On a *trend-following* test set that is fine in bull
markets but loses badly in bear markets — the predictor never flips short. Balancing
the classes re-weights up/down weeks equally, removing that majority bias so the model
can issue genuine short signals when the trend features point down. This is what makes
the resulting long/short strategy adapt to regime instead of being a closet long-only
position — the property that matters on the adversarial 2022 (bear) / 2023 (bull)
test set.

The estimator is deterministic (lbfgs solver); ``random_state`` is accepted and stored
for documentation/reproducibility but does not affect the fit.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


class BalancedLogisticModel:
    def __init__(self, C: float = 1.0, max_iter: int = 2000,
                 class_weight: str = "balanced", solver: str = "lbfgs",
                 random_state: int = 42, **kwargs):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.solver = solver
        self.random_state = random_state
        self.model = None

    def initialize(self):
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            X_test=None, y_test=None, metadata=None, **kwargs):
        # Fit on the training split only (1990-2019). Val (2020-2021) is reserved for
        # model selection and is intentionally NOT used here, mirroring the pipeline's
        # default contract and keeping the leakage story trivial.
        metadata = metadata or {}
        self.model.fit(X_train.values, y_train.values.ravel())
        return None, metadata

    def predict(self, X, y=None, metadata=None, **kwargs):
        metadata = metadata or {}
        preds = self.model.predict(X.values)
        y_pred = pd.DataFrame({"target": preds})
        return X, y_pred, metadata
