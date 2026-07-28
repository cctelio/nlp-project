#!/usr/bin/env python
"""Precompute tokenizer-specific SFT datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import PROMPT_TEMPLATE
from src.data.sft import precompute_sft_splits
from src.utils.config import load_experiment_config
from src.utils.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tokenizer_config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.tokenizer_config, overrides=args.overrides)
    data = config.get("data", {})
    strategy = config.get("tokenizer", {}).get("strategy", "default")
    output_dir = args.output_dir or str(resolve_path(data.get("sft_dir", "results/data/sft")) / strategy)
    paths = precompute_sft_splits(
        train_path=resolve_path(data.get("train_path", "results/data/mol_instructions_description_guided/train.jsonl")),
        validation_path=resolve_path(
            data.get("validation_path", "results/data/mol_instructions_description_guided/validation.jsonl")
        ),
        test_path=resolve_path(data["test_path"]) if data.get("test_path") else None,
        output_dir=resolve_path(output_dir),
        tokenizer_config=config.get("tokenizer", {}),
        prompt_template=data.get("prompt_template") or PROMPT_TEMPLATE,
    )
    print(paths)


if __name__ == "__main__":
    main()
