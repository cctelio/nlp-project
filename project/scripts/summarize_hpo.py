#!/usr/bin/env python
"""Summarize best validation-selected HPO runs and available test metrics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo_glob", default="results/hpo/*_hpo_results.json")
    parser.add_argument("--output", default="results/hpo/best_run_summary.csv")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    for key in (
        "validation/tanimoto_similarity_mean",
        "validation/validity",
        "validation/canonical_exact_match_accuracy",
        "validation/exact_match_accuracy",
        "eval_loss",
        "test/tanimoto_similarity_mean",
        "test/validity",
        "test/canonical_exact_match_accuracy",
        "test/exact_match_accuracy",
    ):
        if key in metrics:
            selected[f"{prefix}{key}"] = metrics[key]
    return selected


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(args.hpo_glob)):
        summary = _read_json(path)
        best = summary.get("best") or {}
        run_dir = Path(best.get("run_dir", ""))
        metrics = best.get("metrics", {})
        test_metrics_path = run_dir / "evaluation" / "test_metrics.json"
        if test_metrics_path.exists():
            metrics = {**metrics, **_read_json(test_metrics_path)}
        row = {
            "hpo_file": path,
            "run_dir": str(run_dir),
            "trial_index": best.get("trial_index"),
            "selection_metric": best.get("metric_name"),
            "selection_value": best.get("metric_value"),
        }
        row.update({f"param/{key}": value for key, value in (best.get("assignment") or {}).items()})
        row.update(_flatten_metrics("", metrics))
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
