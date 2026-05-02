from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration invalide : le YAML doit contenir un mapping.")
    model_name = config.get("model_name", config.get("mode"))
    if not model_name:
        raise ValueError("La configuration doit definir `model_name` ou `mode`.")
    config["model_name"] = model_name
    config["mode"] = model_name
    return config
