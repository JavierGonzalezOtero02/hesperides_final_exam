"""Grid search for the logistic model, selected by walk-forward validation.

Hyperparameters (C regularisation strength, L1/L2 penalty) are ranked by mean
walk-forward Sharpe over the pre-test period (1990-2021) — the OOS test set is
never touched here. Same expanding-window harness as walk_forward_validation.py:
the RobustScaler and the model are re-fit on each fold's training portion only.

Run: uv run python scripts/grid_search_logistic.py
"""

import sys
import pathlib
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")  # silence sklearn deprecation noise

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from code.modules.stages.transforms.dataset_loader import DatasetLoader
from code.modules.stages.transforms.time_series_xy_split import TimeSeriesXYSplit
from code.modules.stages.transforms.robust_scaler import RobustScalerTransform

TEST_START = pd.Timestamp("2022-01-03")  # never seen here
MIN_TRAIN = 520
HORIZON = 52
STEP = 52
TPY = 52

TARGET = "binary_direction_1d"
SMA_WINDOW = 10
FEATURE_GROUPS = ["sma", "momentum", "rsi"]

C_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]


def sharpe(signal, close):
    close = np.asarray(close, dtype=float)
    r = np.diff(close) / close[:-1]
    m = min(len(signal), len(r))
    sr = signal[:m] * r[:m]
    if len(sr) == 0 or sr.std() == 0:
        return 0.0
    return sr.mean() / sr.std() * np.sqrt(TPY)


def main():
    X, _, md = DatasetLoader("data/sp500/processed", "sp500_dataset").transform()
    X, y, md = TimeSeriesXYSplit(TARGET, SMA_WINDOW, FEATURE_GROUPS).transform(X, None, md)

    dates = pd.to_datetime(X["Date"]).values
    close = np.asarray(md["close_prices"], dtype=float)
    feat_cols = [c for c in X.columns if c != "Date"]

    pre = dates < np.datetime64(TEST_START)
    Xp = X.loc[pre, feat_cols].reset_index(drop=True)
    yp = y.loc[pre].reset_index(drop=True)
    closep = close[pre]
    N = len(Xp)
    anchors = list(range(MIN_TRAIN, N - HORIZON + 1, STEP))

    # Pre-scale each fold once (the scaler is config-independent) so the grid only
    # re-fits the cheap logistic per (C, penalty).
    folds = []
    for a in anchors:
        Xtr, ytr = Xp.iloc[:a], yp.iloc[:a]
        Xva = Xp.iloc[a:a + HORIZON]
        sc = RobustScalerTransform()
        sc.fit(Xtr, ytr, {})
        Xtr_s, _, _ = sc.transform(Xtr)
        Xva_s, _, _ = sc.transform(Xva)
        folds.append((Xtr_s.values, ytr.values.ravel(),
                      Xva_s.values, closep[a:a + HORIZON + 1]))

    results = []
    for C in C_GRID:
        scores = []
        for Xtr_v, ytr_v, Xva_v, close_va in folds:
            clf = LogisticRegression(C=C, max_iter=1000)  # default L2 penalty
            clf.fit(Xtr_v, ytr_v)
            signal = 2 * clf.predict(Xva_v) - 1
            scores.append(sharpe(signal, close_va))
        s = np.array(scores)
        results.append((C, s.mean(), s.std(), np.median(s), (s > 0).mean()))

    results.sort(key=lambda r: -r[1])
    print(f"\nGrid search (logistic, L2) — {len(anchors)} walk-forward folds, "
          f"ranked by mean val Sharpe\n")
    print(f"{'C':>10}{'mean':>8}{'std':>8}{'median':>8}{'%>0':>7}")
    print("-" * 41)
    for C, mean, std, med, pct in results:
        print(f"{C:>10}{mean:>8.3f}{std:>8.3f}{med:>8.3f}{pct:>7.0%}")
    best = results[0]
    print(f"\nBest by validation: C={best[0]} (mean val Sharpe {best[1]:.3f})")
    print("NOTE: this val Sharpe is optimistically biased by the search itself; "
          "the honest estimate is the test set, touched once with this config.\n")


if __name__ == "__main__":
    main()
