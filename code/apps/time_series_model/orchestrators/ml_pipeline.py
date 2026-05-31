"""MLPipeline — Orchestrates the full ML pipeline."""

from code.apps.time_series_model.utils import instantiate


class MLPipeline:
    def __init__(self, config: dict):
        self.config = config

    def initialize(self):
        pipeline = self.config["pipeline"]
        self.etl_stages = [instantiate(cfg) for cfg in pipeline["etl"]]
        self.dataset_loader = instantiate(pipeline["dataset_loader"])
        self.xy_split = instantiate(pipeline["xy_split"])
        self.validation_split = instantiate(pipeline["validation_split"])
        self.dpp_stages = [instantiate(cfg) for cfg in pipeline["dpp"]]
        self.model = instantiate(pipeline["model"])
        self.pos_stages = [instantiate(cfg) for cfg in pipeline["pos"]]
        self.metrics = instantiate(pipeline["metrics"])
        self.model_loader = instantiate(pipeline["model_loader"])

    def etl(self):
        for stage in self.etl_stages:
            X, y, metadata = stage.run(None, None, {})

    def train(self):
        metadata = {}
        # Load data
        X, y, metadata = self.dataset_loader.transform(None, None, metadata)
        # XY split
        X, y, metadata = self.xy_split.transform(X, y, metadata)
        # Validation split
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = self.validation_split.transform(X, y, metadata)
        # DPP: fit on train, transform all
        for stage in self.dpp_stages:
            X_train, y_train, metadata = stage.fit(X_train, y_train, metadata)
            X_train, y_train, metadata = stage.transform(X_train, y_train, metadata)
            X_val, y_val, metadata = stage.transform(X_val, y_val, metadata)
            X_test, y_test, metadata = stage.transform(X_test, y_test, metadata)
        # Model
        self.model.initialize()
        self.model.fit(X_train, y_train, X_val, y_val, X_test, y_test, metadata)
        _, y_pred, metadata = self.model.predict(X_test, y_test, metadata)
        # POS
        for stage in self.pos_stages:
            X_test, y_pred, metadata = stage.transform(X_test, y_pred, metadata)
        # Metrics
        score_df, metadata = self.metrics.run(y_test, y_pred, metadata)
        # Save
        model_name = self.config.get("model_name", "model")
        version = self.config.get("experiment_version", "v1")
        models_path = self.config.get("models_path", "models")
        self.model_loader.save(f"{model_name}_{version}", self.model, models_path)
        return score_df

    def predict(self, X=None):
        metadata = {}
        for stage in self.dpp_stages:
            X, _, metadata = stage.transform(X, None, metadata)
        _, y_pred, metadata = self.model.predict(X, None, metadata)
        for stage in self.pos_stages:
            X, y_pred, metadata = stage.transform(X, y_pred, metadata)
        return y_pred

