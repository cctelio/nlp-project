#!/usr/bin/env python
"""Prepare Mol-Instructions description-guided molecule design splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.mol_instructions import prepare_description_guided_splits
from src.utils.config import load_experiment_config
from src.utils.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. data.raw_path=/tmp/data.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.overrides)
    data = config.get("data", {})
    paths = prepare_description_guided_splits(
        output_dir=resolve_path(data.get("processed_dir", "results/data/mol_instructions_description_guided")),
        raw_path=resolve_path(data["raw_path"]) if data.get("raw_path") else None,
        dataset_name=data.get("dataset_name", "zjunlp/Mol-Instructions"),
        archive_filename=data.get("archive_filename", "data/Molecule-oriented_Instructions.zip"),
        cache_dir=resolve_path(data.get("cache_dir", "results/cache")),
        task_filter=data.get("task_filter"),
        train_fraction=float(data.get("train_fraction", 0.8)),
        validation_fraction=float(data.get("validation_fraction", 0.1)),
        seed=int(config.get("seed", 42)),
        max_samples=data.get("max_samples"),
        output_format=data.get("output_format", "auto"),
        canonicalize=bool(data.get("canonicalize_smiles", True)),
    )
    print(paths)


if __name__ == "__main__":
    main()
