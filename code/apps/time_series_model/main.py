"""S&P 500 Forecasting — Entry Point

Final exam pipeline. DO NOT MODIFY this file.
Usage: uv run python -m code.apps.time_series_model.main [etl|train|predict]
"""

import sys


class Main:
    """Facade — delegates all logic to the ML pipeline orchestrator."""

    def __init__(self, config_path: str = "code/apps/time_series_model/configs/time_series_model.yaml"):
        self.config_path = config_path
        self.pipeline = None

    def initialize(self) -> None:
        from code.apps.time_series_model.utils import ConfigLoader
        from code.apps.time_series_model.orchestrators.ml_pipeline import MLPipeline
        config = ConfigLoader.load(self.config_path)
        self.pipeline = MLPipeline(config)
        self.pipeline.initialize()

    def etl(self):
        return self.pipeline.etl()

    def train(self):
        return self.pipeline.train()

    def predict(self, X=None):
        return self.pipeline.predict(X)


if __name__ == "__main__":
    config = "code/apps/time_series_model/configs/time_series_model.yaml"
    main = Main(config)
    main.initialize()

    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode == "etl":
        main.etl()
        print("✓ ETL completed")
    elif mode == "train":
        score_df = main.train()
        print(score_df)
    elif mode == "predict":
        score_df = main.predict()
        print(score_df)
    elif mode is None:
        # No argument → full pipeline: etl → train → predict
        print("=== etl ===")
        main.etl()
        print("=== train ===")
        score_df = main.train()
        print(score_df)
        print("=== predict ===")
        score_df = main.predict()
        print(score_df)
    else:
        print(f"Unknown mode '{mode}'. Use: etl | train | predict")
        sys.exit(1)

