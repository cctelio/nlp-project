#!/usr/bin/env python
"""Print a compact table of training and generation metrics for run directories."""

import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - used only in minimal environments.
    yaml = None


def _read_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml(path):
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _get(mapping, dotted_key, default=""):
    cursor = mapping
    for key in dotted_key.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.4g}".format(value)
    return str(value)


def _row(run_dir):
    train = _read_json(run_dir / "train_metrics.json")
    validation = _read_json(run_dir / "evaluation" / "validation_metrics.json")
    config = _read_yaml(run_dir / "config.yaml")
    return {
        "run": run_dir.name,
        "strategy": train.get("tokenizer_strategy", _get(config, "tokenizer.strategy")),
        "max_len": _get(config, "training.max_length"),
        "steps": _get(config, "training.max_steps"),
        "lr": _get(config, "training.learning_rate"),
        "batch": _get(config, "training.per_device_train_batch_size"),
        "accum": _get(config, "training.gradient_accumulation_steps"),
        "packing": _get(config, "training.packing"),
        "gen_new": _get(config, "generation.max_new_tokens"),
        "train_loss": train.get("train_loss"),
        "eval_loss": train.get("eval_loss"),
        "validity": validation.get("validation/validity"),
        "invalid": validation.get("validation/invalid_rate"),
        "tanimoto": validation.get("validation/tanimoto_similarity_mean"),
        "exact": validation.get("validation/exact_match_accuracy"),
        "empty": validation.get("validation/empty_generation_rate"),
        "gen_len": validation.get("validation/generated_tokenized_smiles_length_mean"),
        "target_len": validation.get("validation/target_tokenized_smiles_length_mean"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default="results/short_runs",
        help="Directory containing run subdirectories.",
    )
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root)
    run_dirs = sorted(
        [path for path in root.glob("*") if (path / "train_metrics.json").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[: args.limit]

    columns = [
        "run",
        "strategy",
        "max_len",
        "steps",
        "lr",
        "batch",
        "accum",
        "packing",
        "gen_new",
        "train_loss",
        "eval_loss",
        "validity",
        "invalid",
        "tanimoto",
        "exact",
        "empty",
        "gen_len",
        "target_len",
    ]
    print("\t".join(columns))
    for run_dir in run_dirs:
        row = _row(run_dir)
        print("\t".join(_fmt(row[column]) for column in columns))


if __name__ == "__main__":
    main()
