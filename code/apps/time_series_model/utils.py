"""Utilities — Config loading and dynamic class instantiation."""

import importlib
from pathlib import Path

import yaml


class ConfigLoader:
    @staticmethod
    def load(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)


def instantiate(cfg: dict):
    """Instantiate a class from a config dict with 'class' and 'params' keys."""
    if not cfg or "class" not in cfg:
        from code.modules.stages.transforms.identity import Identity
        return Identity()

    class_path = cfg["class"]
    params = cfg.get("params", {})

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**params)

