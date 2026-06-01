"""MetricsSharpe — Long/short backtest + Sharpe Ratio evaluation. IMMUTABLE.

This module owns the backtest: it turns the model predictions into a
long/short trading signal, computes the strategy returns over the test set
(using the test close prices) and reports the annualized Sharpe Ratio.
"""

import numpy as np
import pandas as pd


class MetricsSharpe:
    def __init__(self, trading_days_per_year: int = 252,
                 signal_mode: str = "classification", **kwargs):
        self.trading_days_per_year = trading_days_per_year
        self.signal_mode = signal_mode

    def run(self, y_true, y_pred, metadata=None, **kwargs):
        metadata = metadata or {}

        # --- Predictions → long/short signal --------------------------- #
        if isinstance(y_pred, pd.DataFrame):
            preds = y_pred.values.flatten()
        else:
            preds = np.array(y_pred).flatten()

        if self.signal_mode == "classification":
            # 1 → +1 (long), 0 → -1 (short)
            signal = 2 * preds - 1
        elif self.signal_mode == "regression":
            # sign of the predicted return
            signal = np.sign(preds)
        else:
            raise ValueError(f"Unknown signal_mode: {self.signal_mode}")

        # --- Backtest over the test set -------------------------------- #
        close_prices_test = metadata.get("close_prices_test")
        if close_prices_test is None:
            raise ValueError("close_prices_test not found in metadata.")
        close = np.asarray(close_prices_test, dtype=float)
        real_returns = np.diff(close) / close[:-1]
        # signal[t] applies to the return from t to t+1
        min_len = min(len(signal), len(real_returns))
        sr = signal[:min_len] * real_returns[:min_len]
        metadata["strategy_returns"] = sr

        # Sharpe Ratio
        if sr.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (sr.mean() / sr.std()) * np.sqrt(self.trading_days_per_year)

        # Annualized return
        annualized_return = sr.mean() * self.trading_days_per_year

        # Max drawdown
        cumulative = (1 + sr).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Win rate
        win_rate = (sr > 0).sum() / len(sr) if len(sr) > 0 else 0.0

        score_df = pd.DataFrame([{
            "sharpe_ratio": round(sharpe_ratio, 4),
            "annualized_return": round(annualized_return, 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
        }])

        return score_df, metadata

