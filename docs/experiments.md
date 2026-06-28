# Experiment Log — S&P 500 Weekly Direction

Record of the modelling experiments run on top of the baseline, with results and
the methodology used to keep them honest. Metric is the annualised **Sharpe Ratio**
of the long/short strategy produced by `MetricsSharpe` (the immutable judge).

## Fixed setup

- **Data:** weekly OHLCV `^GSPC`, 1990–2023 (Yahoo Finance).
- **Splits (immutable):** Train 1990→2019 (~1506 bars) · Val 2020→2021 (~104) ·
  Test OOS 2022-01-03→2023-12-29 (~104).
- **Target:** `binary_direction_1d` (1 = next week up, 0 = down).
- **Judge:** `MetricsSharpe`, classification mode (`1→+1` long, `0→-1` short),
  annualised with √52. `y_true` is not used — direction comes from the prediction,
  realised return from the test `Close` prices.
- **Golden rule:** model/hyperparameter selection is done on **validation**, never
  on the test set. The test set is read once at the end to report the final number.

---

## Results at a glance

| # | Experiment | Selection metric | Result |
|---|---|---|---|
| 0 | Baseline (logistic, `sma` only, no scaling) | — | **Test Sharpe 0.1469** |
| 1 | Feature expansion + RobustScaler (logistic) | — | **Test Sharpe 0.5505** (train == predict) |
| 2 | Model comparison on test (anti-pattern) | test (wrong) | HGB 0.74 > Logistic 0.55 > SVM 0.29 > RF 0.11 |
| 3 | Single-split validation vs test | val (2020–21) | Logistic best on val (1.40) |
| 4 | Walk-forward validation (20 folds) | mean val Sharpe | **Logistic 0.698** > SVM 0.670 > RF 0.500 > HGB 0.347 |
| 5 | Grid search logistic `C` (walk-forward) | mean val Sharpe | `C` ≈ insensitive; nominal best `C=0.001` (0.785) |

---

## Experiment 0 — Baseline

Logistic regression on a single feature (`sma_10` of weekly returns), no
preprocessing. This is the reference to beat.

```
Test Sharpe 0.1469 · AnnRet +2.79% · MaxDD -23.16% · Win 50.49%
```

## Experiment 1 — Feature expansion + RobustScaler

Two changes via the YAML levers:
- `xy_split.feature_groups`: `["sma"]` → `["sma", "momentum", "rsi"]` (1 → 8 features).
- `dpp`: `Identity` → `RobustScalerTransform` (median/IQR scaling, fit on train only).

```
Test Sharpe 0.5505 · AnnRet +10.42% · MaxDD -19.28% · Win 53.40%
```

`train` and `predict` report the identical 0.5505 → the saved bundle is reproducible
(delivery validity criterion met).

**Leakage audit (verified in code, not assumed):**
- RobustScaler `center_` equals the **train** median exactly (diff 0.0) and not the
  full-dataset median (diff 0.51) → fit on train only, no scaling leakage.
- All features are backward-looking: recomputing them on a series truncated at `t`
  yields identical values to the full series → no future information at `t`.
- `target` (uses `shift(-1)`) is the label, kept out of the feature matrix.
- Backtest alignment: `signal[t]` (info up to `t`) multiplies the realised `t→t+1`
  return. No look-ahead.

## Experiment 2 — Model comparison ON TEST (anti-pattern, kept as a lesson)

Same features + scaler, swapping the model, scored on the **test** set:

| Model | Test Sharpe |
|---|---|
| HistGradBoost | 0.7396 |
| Logistic | 0.5505 |
| SVM-RBF | 0.2886 |
| RandomForest | 0.1069 |

Picking the winner here (HGB) would mean selecting on the test set — invalid. See
Experiment 4 for why this ranking is misleading.

## Experiment 3 — Single-split validation (2020–2021)

Scoring the same models on the held-out validation window and only then peeking at
test:

| Model | Val Sharpe | Test Sharpe |
|---|---|---|
| Logistic | 1.4022 | 0.5505 |
| RandomForest | 1.0682 | 0.1069 |
| SVM-RBF | 0.9794 | 0.2886 |
| HistGradBoost | 0.5312 | 0.7396 |

By validation, the **logistic wins**. Note the val Sharpes are inflated (>1): the
2020–2021 bull recovery rewards long-biased models, so a single short window is a
fragile basis for selection — motivating Experiment 4.

## Experiment 4 — Walk-forward validation (20 folds, 1990–2021)

Expanding-window walk-forward over the pre-test period (≥520-week train, 52-week
forward slice per fold). RobustScaler and model re-fit on each fold's train only.
Test set never touched. Script: `scripts/walk_forward_validation.py`.

| Model | mean | std | median | %>0 |
|---|---|---|---|---|
| **Logistic** | **0.698** | 1.086 | 0.717 | 70% |
| SVM-RBF | 0.670 | 1.023 | 0.644 | 65% |
| RandomForest | 0.500 | 0.887 | 0.457 | 60% |
| HistGradBoost | 0.347 | 0.902 | 0.501 | 70% |

**Key findings:**
- **Logistic is the robust winner** (best mean and median, positive in 70% of years).
- **HistGradBoost is unmasked:** best on test (0.74) but worst on robust validation
  (0.347). Its test result was luck/test-overfit, not skill — the exact trap the
  validation-first rule prevents.
- **SVM-RBF justifies the RobustScaler** (2nd place); the scaler was not decorative,
  though the simpler linear model still wins.
- **Honest caveat:** `std` (~1.0) > mean (~0.7) for every model. Weekly index
  direction is mostly noise; no model has a strong edge. The ranking is real, but
  absolute confidence is low.

## Experiment 5 — Grid search for the logistic (`C`)

Grid over the L2 regularisation strength `C`, ranked by mean walk-forward Sharpe.
Script: `scripts/grid_search_logistic.py`.

| C | mean | std | median | %>0 |
|---|---|---|---|---|
| 0.001 | 0.785 | 1.209 | 0.809 | 70% |
| 2.0 | 0.721 | 1.068 | 0.717 | 70% |
| 1.0 | 0.698 | 1.086 | 0.717 | 70% |
| 10.0 | 0.698 | 1.051 | 0.717 | 70% |
| 0.05 | 0.691 | 1.178 | 0.547 | 70% |
| 0.01 | 0.691 | 1.159 | 0.669 | 65% |
| 0.5 | 0.658 | 1.126 | 0.717 | 65% |
| 100.0 | 0.656 | 1.069 | 0.717 | 70% |
| 0.1 | 0.651 | 1.212 | 0.613 | 60% |

**Findings:**
- `C` is **barely a lever**: the whole range spans 0.65–0.79, well inside the ~1.1
  per-fold std. The logistic is stable and feature-driven, not regularisation-driven.
- Nominal best `C=0.001` (strong shrinkage) leads on mean, median and %>0, with a
  principled story (heavy regularisation helps in a noisy problem) — but the gain is
  within noise, and topping a 9-way search inflates the validation figure.

---

## Current state and open decision

- **Selected model (by robust validation): Logistic regression** on
  `["sma", "momentum", "rsi"]` with RobustScaler — best walk-forward Sharpe and the
  simplest (Occam). This is what the YAML currently points to.
- **Open hyperparameter choice:**
  - `C=1.0` (current): zero search-overfit, clean story; saved bundle already gives
    test 0.5505.
  - `C=0.001`: validation-optimal with a regularisation rationale, but the
    improvement is within noise; would require retrain + a single final test read.
- **Final step (pending):** read the test set **once** with the chosen config to
  report the delivery Sharpe.

## Rejected / deprioritised directions

- **ARIMA + GARCH:** for a binary direction signal, GARCH variance does not change
  the sign of the conditional mean; sparse sentiment-style history and reproducibility
  concerns. Deprioritised.
- **Sentiment analysis:** no point-in-time history for the 1990–2010 train span,
  high look-ahead risk, non-deterministic ETL → not reproducible. Replaced by the
  idea of price-based "alternative data" (e.g. VIX) if revisited.

## Reproduce

```bash
uv run python -m code.apps.time_series_model.main etl     # rebuild table.csv + test.csv
uv run python -m code.apps.time_series_model.main train   # fit + report test Sharpe
uv run python -m code.apps.time_series_model.main predict # reproduce from saved bundle
uv run python scripts/walk_forward_validation.py          # robust model ranking
uv run python scripts/grid_search_logistic.py             # logistic C grid
```
