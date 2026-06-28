"""Grid search for every candidate model, selected by walk-forward validation.

Tuning only the logistic biases the model comparison. This script grids each
model family (Logistic, SVM-RBF, RandomForest, HistGradBoost) on the same
expanding-window walk-forward harness over the pre-test period (1990-2021) and
reports the best config per family plus a best-of-each ranking. The OOS test set
is never touched here.

Caveat: searching many configs against the same noisy validation inflates the
winning Sharpe (multiple testing). Grids are kept modest; the honest estimate of
the chosen config is still the test set, read once at the end.

Run: uv run python scripts/grid_search_models.py
"""

import sys
import pathlib
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from code.modules.stages.transforms.dataset_loader import DatasetLoader
from code.modules.stages.transforms.time_series_xy_split import TimeSeriesXYSplit
from code.modules.stages.transforms.robust_scaler import RobustScalerTransform

TEST_START = pd.Timestamp("2022-01-03")
MIN_TRAIN = 520
HORIZON = 52
STEP = 52
TPY = 52

TARGET = "binary_direction_1d"
SMA_WINDOW = 10
FEATURE_GROUPS = ["sma", "momentum", "rsi"]


def sharpe(signal, close):
    close = np.asarray(close, dtype=float)
    r = np.diff(close) / close[:-1]
    m = min(len(signal), len(r))
    sr = signal[:m] * r[:m]
    if len(sr) == 0 or sr.std() == 0:
        return 0.0
    return sr.mean() / sr.std() * np.sqrt(TPY)


def build_grids():
    """Return {model_name: [(label, estimator_factory), ...]}."""
    grids = {}

    grids["Logistic"] = [
        (f"C={c}", (lambda c=c: LogisticRegression(C=c, max_iter=1000)))
        for c in [0.001, 0.01, 0.1, 1.0, 10.0]
    ]

    grids["SVM-RBF"] = [
        (f"C={c},g={g}", (lambda c=c, g=g: SVC(C=c, kernel="rbf", gamma=g)))
        for c in [0.5, 1.0, 2.0, 10.0]
        for g in ["scale", 0.01, 0.1]
    ]

    grids["RandomForest"] = [
        (f"depth={d},leaf={l}",
         (lambda d=d, l=l: RandomForestClassifier(
             n_estimators=400, max_depth=d, min_samples_leaf=l,
             max_features="sqrt", random_state=42, n_jobs=-1)))
        for d in [3, 4, 5, 8]
        for l in [10, 30, 50]
    ]

    grids["HistGradBoost"] = [
        (f"lr={lr},leaves={ml},leaf={msl}",
         (lambda lr=lr, ml=ml, msl=msl: HistGradientBoostingClassifier(
             learning_rate=lr, max_leaf_nodes=ml, min_samples_leaf=msl,
             l2_regularization=1.0, max_iter=300, early_stopping=True,
             n_iter_no_change=20, validation_fraction=0.15, random_state=42)))
        for lr in [0.02, 0.05, 0.1]
        for ml in [7, 15, 31]
        for msl in [20, 50]
    ]

    return grids


def score_config(factory, folds):
    scores = []
    for Xtr, ytr, Xva, close_va in folds:
        clf = factory()
        clf.fit(Xtr, ytr)
        signal = 2 * clf.predict(Xva) - 1
        scores.append(sharpe(signal, close_va))
    return np.array(scores)


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

    # Pre-scale folds once (RobustScaler is model-independent).
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

    grids = build_grids()
    best_per_model = {}

    for model_name, configs in grids.items():
        ranked = []
        for label, factory in configs:
            s = score_config(factory, folds)
            ranked.append((label, s.mean(), s.std(), np.median(s), (s > 0).mean()))
        ranked.sort(key=lambda r: -r[1])
        best_per_model[model_name] = ranked[0]

        print(f"\n=== {model_name} ({len(configs)} configs) — top 5 by mean val Sharpe ===")
        print(f"{'config':<26}{'mean':>8}{'std':>8}{'median':>8}{'%>0':>7}")
        print("-" * 57)
        for label, mean, std, med, pct in ranked[:5]:
            print(f"{label:<26}{mean:>8.3f}{std:>8.3f}{med:>8.3f}{pct:>7.0%}")

    print("\n\n##### BEST-OF-EACH (walk-forward, tuned fairly) #####")
    print(f"{'model':<16}{'best config':<26}{'mean':>8}{'std':>8}{'%>0':>7}")
    print("-" * 65)
    for model_name, (label, mean, std, med, pct) in sorted(
            best_per_model.items(), key=lambda kv: -kv[1][1]):
        print(f"{model_name:<16}{label:<26}{mean:>8.3f}{std:>8.3f}{pct:>7.0%}")
    print()


if __name__ == "__main__":
    main()
