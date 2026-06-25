# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A take-home final exam: build an ML model that forecasts the **weekly direction of the S&P 500** and aims to maximize the **Sharpe Ratio** of the resulting long/short trading strategy over a fixed, out-of-sample test set (2022–2023). The README (in Spanish) is the authoritative spec.

**How it's graded (the real criteria).** The practical part is judged by correct execution and the value of the evaluation metric. There are **3 strict pass/fail criteria** — all three must hold:

1. **Runs first-try from the provided `main.py`.** The code must perform inference on the test set and report the target metrics on the *first* attempt, inside the environment installed from `pyproject.toml`. No manual fix-ups, no missing deps.
2. **Beats the baseline metric.** The result must be **strictly superior to the provided baseline** (baseline Sharpe ≈ **0.1469**). Beating it is **required**, not optional — a low Sharpe is *not* rescued by a good explanation.
3. **No data leakage involving the test set.** This is **critical**. Any leakage of test information into training fails the submission.

Beyond pass/fail: submissions are **semi-randomly audited**, with deliberate focus on **extraordinarily high or negative** results, to trace the root cause of any problem and to rule out irresponsible/uncritical use of AI. So an extreme Sharpe (suspiciously high *or* very negative) will be scrutinized — you must be able to **justify how it arose** and demonstrate it is not an artifact of leakage or a bug. Sound, well-argued reasoning still matters, but it sits *on top of* meeting the three hard criteria, not in place of them.

## Commands

Dependencies are managed with `uv` (reads `uv.lock`, creates `.venv/` automatically). Always run via `uv run`.

```bash
uv sync                                                      # install deps

# Pipeline modes (the only entry point):
uv run python -m code.apps.time_series_model.main etl        # download data, persist fixed test set
uv run python -m code.apps.time_series_model.main train      # fit model, save bundle, print Sharpe
uv run python -m code.apps.time_series_model.main predict    # load saved bundle, infer on test.csv, print Sharpe
uv run python -m code.apps.time_series_model.main            # full sequence: etl → train → predict

./scripts/build_project.sh                                   # build wheel into dist/ (uses uv build)
```

Dev tooling is declared in `pyproject.toml` `[dev]` (pytest, black, ruff, mypy; line-length 100). The `tests/` tree exists but currently holds only `.gitkeep` files — there are no tests yet. Run a single test with `uv run pytest tests/path::test_name`.

**Deliverable check:** the submission is valid only if `predict` runs without errors, loads the saved model, and prints a Sharpe. Always confirm `train` then `predict` reproduce the same number.

## Architecture

**Slots & Protocols pipeline.** `MLPipeline` (`orchestrators/ml_pipeline.py`) defines a fixed sequence of stages; each stage is any object satisfying a duck-typed protocol. Stages are not imported directly — `utils.instantiate()` builds them dynamically from `{class, params}` dicts in the YAML config. **Changing the model, features, or hyperparameters is done entirely in YAML, not Python.**

The single config file is `code/apps/time_series_model/configs/time_series_model.yaml`. The data contract passed between every stage is the tuple `(X, y, metadata)`, where `metadata` is a dict carrying auxiliary data (notably `close_prices_test`, needed by the backtest).

Stage order (see `ml_pipeline.py`):
```
YahooFinanceETL → DatasetLoader → TimeSeriesXYSplit → TemporalValidationSplit
  → DPP stages (fit on train, transform train/val/test) → Model → MetricsSharpe
```

- **TimeSeriesXYSplit** builds features and target. `target` ∈ {`binary_direction_1d` (classification), `pct_return_1d`, `log_return_1d` (regression)}. `feature_groups` selects feature families (sma, momentum, rsi, volatility, bollinger, volume, calendar) — baseline is `["sma"]` only.
- **TemporalValidationSplit** splits by fixed dates (val 2020, test 2022-01-03+). These dates are immutable.
- **MetricsSharpe** converts predictions → long/short signal (classification: 1→long, 0→short; regression: `sign(y_pred)`), backtests, and annualizes Sharpe by √(trading_days_per_year=52). If you switch to a regression target, set `metrics.params.signal_mode: "regression"`.
- **train** saves a bundle `{"model": ..., "dpp": [...]}` via `BasicModelLoader` to `models/sp500/<model_name>_<experiment_version>/pipeline.pkl`. **predict** reloads that bundle plus the committed `test.csv` — it does NOT rebuild features, so the fitted DPP stages must be in the bundle.

### Stage protocols (when adding new modules)

Models (`code/modules/stages/models/`):
```python
def initialize(self): ...                                          # build the internal estimator
def fit(self, X_train, y_train, X_val=None, y_val=None,
        X_test=None, y_test=None, metadata=None, **kwargs): return None, metadata
def predict(self, X, y=None, metadata=None, **kwargs):
    return X, y_pred, metadata   # y_pred is pd.DataFrame with a "target" column
```

Transforms / DPP stages (`code/modules/stages/transforms/`):
```python
def fit(self, X, y=None, metadata=None, **kwargs): return X, y, metadata
def transform(self, X, y=None, metadata=None, **kwargs): return X, y, metadata
```

Register either in the YAML (`model:` is a single block; `dpp:` is a list, chained in order).

## Hard constraints (exam rules)

Do NOT modify these — doing so invalidates the exam:
- `code/apps/time_series_model/main.py` — immutable entry point
- `code/modules/stages/metrics/metrics_sharpe.py` — the judge
- `data/sp500/processed/sp500_dataset/test/test.csv` — the fixed OOS test set
- The split dates in `temporal_validation_split.py` / the YAML (`val_start_date`, `test_start_date`)

Allowed levers: swap the `model`, add `dpp` preprocessing stages, change `target`/`sma_window`/`feature_groups`, extend the ETL `start_date`, add new module files, add dependencies to `pyproject.toml`.
