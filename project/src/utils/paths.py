"""Path and run-directory helpers."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the repository root unless it is absolute."""
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return (root or REPO_ROOT) / value


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value


def timestamp_run_id() -> str:
    """Return a compact timestamp suitable for run directory names."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_run_dir(output_root: str | Path, strategy: str, run_name: str | None = None) -> Path:
    """Create a strategy-specific run directory."""
    run_id = run_name or os.environ.get("WANDB_RUN_ID") or timestamp_run_id()
    safe_strategy = strategy.replace("/", "-").replace(" ", "-")
    safe_run_id = run_id.replace("/", "-").replace(" ", "-")
    run_dir = Path(output_root) / f"{safe_strategy}-{safe_run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
