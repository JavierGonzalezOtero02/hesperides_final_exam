"""BasicModelLoader — Pickle-based model persistence."""

import os
import pickle
from pathlib import Path


class BasicModelLoader:
    def __init__(self, **kwargs):
        pass

    def save(self, model_name: str, model, path: str, **kwargs) -> None:
        output_dir = Path(path) / model_name
        os.makedirs(output_dir, exist_ok=True)
        filepath = output_dir / "pipeline.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"  Model saved to {filepath}")

    def load(self, model_name: str, path: str, **kwargs):
        filepath = Path(path) / model_name / "pipeline.pkl"
        with open(filepath, "rb") as f:
            return pickle.load(f)

