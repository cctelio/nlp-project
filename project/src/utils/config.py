"""YAML config loading and lightweight override handling."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load experiment configs. Install with `pip install pyyaml`.") from exc
    return yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    yaml = _require_yaml()
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a YAML mapping."""
    yaml = _require_yaml()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries and return a new object."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_override(value: str) -> tuple[list[str], Any]:
    """Parse a CLI override of the form `a.b.c=value`."""
    yaml = _require_yaml()
    if "=" not in value:
        raise ValueError(f"Override must be key=value, got: {value}")
    key, raw_value = value.split("=", 1)
    path = [part for part in key.split(".") if part]
    if not path:
        raise ValueError(f"Override key is empty: {value}")
    return path, yaml.safe_load(raw_value)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted key overrides to a config mapping."""
    result = copy.deepcopy(config)
    for override in overrides or []:
        path, parsed_value = parse_override(override)
        cursor = result
        for key in path[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = parsed_value
    return result


def load_experiment_config(
    config_path: str | Path,
    tokenizer_config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load base config, merge tokenizer config, then apply CLI overrides."""
    config = load_yaml(config_path)
    if tokenizer_config_path:
        tokenizer_config = load_yaml(tokenizer_config_path)
        config = deep_update(config, tokenizer_config)
    return apply_overrides(config, overrides)
