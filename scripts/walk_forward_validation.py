"""Walk-forward validation over the pre-test period (1990-2021).

Model selection must NOT touch the OOS test set (2022-2023). This script ranks
candidate models with an expanding-window walk-forward over the train+val span:
for each fold it re-fits the RobustScaler and the model on that fold's training
portion only (no leakage) and scores the Sharpe Ratio on the held-out forward
slice. Reporting mean/std/median across folds gives a regime-robust ranking,
unlike a single validation split dominated by the 2020-2021 bull recovery.

Run: uv run python scripts/walk_forward_validation.py
"""

import sys
import pathlib

# Ensure the repo root is importable regardless of how the script is launched.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from code.modules.stages.transforms.dataset_loader import DatasetLoader
from code.modules.stages.transforms.time_series_xy_split import TimeSeriesXYSplit
from code.modules.stages.transforms.robust_scaler import RobustScalerTransform
from code.modules.stages.models.logistic_baseline import LogisticBaselineModel
from code.modules.stages.models.hist_gradient_boosting import HistGradientBoostingModel
from code.modules.stages.models.random_forest import RandomForestModel
from code.modules.stages.models.svm_rbf import SVMRBFModel

TEST_START = pd.Timestamp("2022-01-03")  # test set is never seen here
MIN_TRAIN = 520   # ~10 years of weekly bars before the first validation fold
HORIZON = 52      # 1-year forward validation slice per fold
STEP = 52         # non-overlapping folds
TPY = 52          # Sharpe annualisation factor (weekly bars)

TARGET = "binary_direction_1d"
SMA_WINDOW = 10
FEATURE_GROUPS = ["sma", "momentum", "rsi"]


def sharpe(signal, close):
    """Annualised Sharpe of a long/short signal, same convention as MetricsSharpe."""
    close = np.asarray(close, dtype=float)
    r = np.diff(close) / close[:-1]
    m = min(len(signal), len(r))
    sr = signal[:m] * r[:m]
    if len(sr) == 0 or sr.std() == 0:
        return 0.0
    return sr.mean() / sr.std() * np.sqrt(TPY)


def build_models():
    # Factories so every fold gets a fresh, unfitted model.
    return {
        "Logistic": lambda: LogisticBaselineModel(C=1.0, max_iter=1000),
        "HistGradBoost": lambda: HistGradientBoostingModel(),
        "RandomForest": lambda: RandomForestModel(),
        "SVM-RBF": lambda: SVMRBFModel(),
    }


def main():
    # Features are backward-looking, so computing them once on the full series is
    # safe (a feature at t depends only on prices up to t).
    X, _, md = DatasetLoader("data/sp500/processed", "sp500_dataset").transform()
    X, y, md = TimeSeriesXYSplit(TARGET, SMA_WINDOW, FEATURE_GROUPS).transform(X, None, md)

    dates = pd.to_datetime(X["Date"]).values
    close = np.asarray(md["close_prices"], dtype=float)
    feat_cols = [c for c in X.columns if c != "Date"]

    # Restrict to the pre-test period — the OOS test set is excluded entirely.
    pre = dates < np.datetime64(TEST_START)
    Xp = X.loc[pre, feat_cols].reset_index(drop=True)
    yp = y.loc[pre].reset_index(drop=True)
    closep = close[pre]
    N = len(Xp)

    anchors = list(range(MIN_TRAIN, N - HORIZON + 1, STEP))
    models = build_models()
    results = {name: [] for name in models}

    for a in anchors:
        Xtr, ytr = Xp.iloc[:a], yp.iloc[:a]
        Xva = Xp.iloc[a:a + HORIZON]
        close_va = closep[a:a + HORIZON + 1]  # +1 so the last signal has a return

        # RobustScaler fit on this fold's train only, then applied to the slice.
        sc = RobustScalerTransform()
        sc.fit(Xtr, ytr, {})
        Xtr_s, _, _ = sc.transform(Xtr)
        Xva_s, _, _ = sc.transform(Xva)

        for name, make in models.items():
            m = make()
            m.initialize()
            m.fit(Xtr_s, ytr, Xva_s, None, Xva_s, None, {})
            pred = m.predict(Xva_s)[1].values.ravel()
            signal = 2 * pred - 1  # classification → long/short
            results[name].append(sharpe(signal, close_va))

    print(f"\nWalk-forward validation — {len(anchors)} folds "
          f"(expanding train ≥{MIN_TRAIN}w, {HORIZON}w forward each)\n")
    print(f"{'MODEL':<16}{'mean':>8}{'std':>8}{'median':>8}{'%>0':>7}")
    print("-" * 47)
    ranking = sorted(results.items(), key=lambda kv: -np.mean(kv[1]))
    for name, scores in ranking:
        s = np.array(scores)
        print(f"{name:<16}{s.mean():>8.3f}{s.std():>8.3f}"
              f"{np.median(s):>8.3f}{(s > 0).mean():>7.0%}")
    print()


if __name__ == "__main__":
    main()
