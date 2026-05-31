"""MetricsSharpe — Sharpe Ratio evaluation. IMMUTABLE."""

import numpy as np
import pandas as pd


class MetricsSharpe:
    def __init__(self, trading_days_per_year: int = 252, **kwargs):
        self.trading_days_per_year = trading_days_per_year

    def run(self, y_true, y_pred, metadata=None, **kwargs):
        metadata = metadata or {}
        strategy_returns = metadata.get("strategy_returns")

        if strategy_returns is None:
            raise ValueError("strategy_returns not found in metadata. Run TradingSignalPOS first.")

        sr = np.array(strategy_returns)

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

