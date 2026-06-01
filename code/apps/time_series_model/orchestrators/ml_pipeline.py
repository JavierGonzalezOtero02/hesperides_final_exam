"""MLPipeline — Orchestrates the full ML pipeline."""

from pathlib import Path

import pandas as pd

from code.apps.time_series_model.utils import instantiate


class MLPipeline:
    """Orchestrates the end-to-end ML pipeline.
    
    Coordinates ETL, feature engineering, temporal splitting, preprocessing,
    model training, and Sharpe Ratio evaluation over a fixed OOS test set
    (2022-01-03 to 2023-12-29). The pipeline is immutable; modules are
    pluggable via YAML configuration.
    """

    def __init__(self, config: dict):
        self.config = config

    def initialize(self):
        """Instantiate all pipeline modules from the config YAML.
        
        Loads and creates:
          - ETL stages: data download
          - Dataset loader: CSV loading
          - XY-split: feature/target engineering
          - Validation split: fixed temporal split (train/val/test)
          - DPP stages: preprocessing (fit on train, apply to all)
          - Model: the ML algorithm
          - Metrics: long/short backtest + Sharpe evaluation
          - Model loader: serialization/persistence
        """
        pipeline = self.config["pipeline"]
        self.etl_stages = [instantiate(cfg) for cfg in pipeline["etl"]]
        self.dataset_loader = instantiate(pipeline["dataset_loader"])
        self.xy_split = instantiate(pipeline["xy_split"])
        self.validation_split = instantiate(pipeline["validation_split"])
        self.dpp_stages = [instantiate(cfg) for cfg in pipeline["dpp"]]
        self.model = instantiate(pipeline["model"])
        self.metrics = instantiate(pipeline["metrics"])
        self.model_loader = instantiate(pipeline["model_loader"])

    def etl(self):
        """Download raw OHLCV data and persist the fixed OOS test set.
        
        Sequence:
          1. Download S&P 500 data via Yahoo Finance → table.csv
          2. Build features (SMA) and target (market direction) → X, y
          3. Perform fixed temporal split by date (train/val/test)
          4. Save the test set to data/sp500/processed/sp500_dataset/test/test.csv
             (committed for reproducibility)
        """
        # Download raw OHLCV and persist the full table
        for stage in self.etl_stages:
            stage.run(None, None, {})

        # Build features/target and perform the fixed temporal split
        metadata = {}
        X, y, metadata = self.dataset_loader.transform(None, None, metadata)
        X, y, metadata = self.xy_split.transform(X, y, metadata)
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = self.validation_split.transform(X, y, metadata)

        # Persist the OOS test set (features + target + close prices for backtest)
        test_df = X_test.copy()
        test_df["target"] = y_test.values.ravel()
        test_df["Close"] = metadata["close_prices_test"][: len(test_df)]
        test_path = Path(self.config["data_path"]) / "processed" / self.config["dataset_name"] / "test" / "test.csv"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(test_path, index=False)
        print(f"  Test set saved to {test_path} ({len(test_df)} rows)")

    def train(self):
        """Train the model on the fixed temporal split and report Sharpe.
        
        Sequence:
          1. Load data → build X/y → temporal split (no randomization; dates fixed)
          2. DPP preprocessing: fit on train, apply to train/val/test
          3. Model: fit on train (with val for early stopping), predict on test
          4. Evaluate: long/short backtest (y_pred → signal) + Sharpe Ratio
             (metrics module owns all backtest logic)
          5. Save the fitted model + DPP bundle for inference reproducibility
        
        Returns:
          DataFrame with columns: sharpe_ratio, annualized_return, max_drawdown, win_rate
        """
        # Load data → build X/y → temporal split
        metadata = {}
        X, y, metadata = self.dataset_loader.transform(None, None, metadata)
        X, y, metadata = self.xy_split.transform(X, y, metadata)
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = self.validation_split.transform(X, y, metadata)

        # DPP: fit on train only, transform all splits
        for stage in self.dpp_stages:
            X_train, y_train, metadata = stage.fit(X_train, y_train, metadata)
            X_train, y_train, metadata = stage.transform(X_train, y_train, metadata)
            X_val, y_val, metadata = stage.transform(X_val, y_val, metadata)
            X_test, y_test, metadata = stage.transform(X_test, y_test, metadata)

        # Model: fit on train, predict on test
        self.model.initialize()
        self.model.fit(X_train, y_train, X_val, y_val, X_test, y_test, metadata)
        _, y_pred, metadata = self.model.predict(X_test, y_test, metadata)

        # Metrics: converts y_pred → long/short signal, computes strategy returns,
        # reports annualized Sharpe + other stats
        score_df, metadata = self.metrics.run(y_test, y_pred, metadata)

        # Save the fitted model + DPP stages as a single deployable bundle
        model_name = self.config.get("model_name", "model")
        version = self.config.get("experiment_version", "v1")
        models_path = self.config.get("models_path", "models")
        bundle = {"model": self.model, "dpp": self.dpp_stages}
        self.model_loader.save(f"{model_name}_{version}", bundle, models_path)
        return score_df

    def predict(self, X=None):
        """Inference over the saved OOS test set; reports Sharpe Ratio.
        
        Loads the trained model bundle and the committed test set, runs
        inference and evaluates the long/short strategy over the test period.
        Ensures reproducible Sharpe across runs since test set is fixed.
        
        Returns:
          DataFrame with columns: sharpe_ratio, annualized_return, max_drawdown, win_rate
        """
        # Load the fitted model bundle (model + fitted DPP stages)
        model_name = self.config.get("model_name", "model")
        version = self.config.get("experiment_version", "v1")
        models_path = self.config.get("models_path", "models")
        bundle = self.model_loader.load(f"{model_name}_{version}", models_path)
        model = bundle["model"]
        dpp_stages = bundle.get("dpp", self.dpp_stages)

        # Load the committed OOS test set (persisted by etl/train)
        test_path = Path(self.config["data_path"]) / "processed" / self.config["dataset_name"] / "test" / "test.csv"
        test_df = pd.read_csv(test_path)

        # Prepare metadata (close prices) + split features/target
        metadata = {"close_prices_test": test_df["Close"].values}
        feature_cols = [c for c in test_df.columns if c not in ("target", "Close")]
        X_test = test_df[feature_cols].copy()
        y_test = test_df[["target"]].copy()

        # DPP transform → model inference → backtest + Sharpe
        for stage in dpp_stages:
            X_test, y_test, metadata = stage.transform(X_test, y_test, metadata)
        _, y_pred, metadata = model.predict(X_test, y_test, metadata)
        score_df, metadata = self.metrics.run(y_test, y_pred, metadata)
        return score_df
