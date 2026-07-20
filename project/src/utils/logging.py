"""Simple logging and artifact serialization helpers."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging for scripts and return a project logger."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("smollm_chem")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON Lines rows from disk."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _csv_cell(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _csv_cell(value) for key, value in row.items()} for row in rows])


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read dictionaries from CSV."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL or CSV rows based on file suffix."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".csv":
        return read_csv(path)
    raise ValueError(f"Unsupported row file extension for {path}; expected .jsonl or .csv")
