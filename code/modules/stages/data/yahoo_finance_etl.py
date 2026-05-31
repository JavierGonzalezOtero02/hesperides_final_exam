"""Yahoo Finance ETL — Downloads S&P 500 OHLCV data."""

import os
from pathlib import Path
import yfinance as yf
import pandas as pd


class YahooFinanceETL:
    def __init__(self, ticker: str = "^GSPC", start_date: str = "2010-01-01",
                 end_date: str = "2023-12-31", dataset_name: str = "sp500_dataset",
                 data_path: str = "data/sp500/processed", **kwargs):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)

    def run(self, X=None, y=None, metadata=None, **kwargs):
        metadata = metadata or {}
        # Download data
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Ensure index is named Date
        if df.index.name != "Date":
            df.index.name = "Date"
        # Reset index to have Date as column
        df = df.reset_index()
        # Keep only OHLCV + Date
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        # Persist
        output_path = self.data_path / self.dataset_name / "table"
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(output_path / "table.csv", index=False)
        print(f"  ETL: {len(df)} rows saved to {output_path / 'table.csv'}")
        return X, y, metadata
