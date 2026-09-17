"""YAML configuration loading with a simple base + override merge."""

from pathlib import Path
from typing import Any

import yaml


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load one or more YAML files, later files overriding earlier ones (shallow-merged per top-level key)."""
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path) as f:
            content = yaml.safe_load(f) or {}
        for key, value in content.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged
