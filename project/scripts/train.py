#!/usr/bin/env python
"""Run one SmolLM chemical tokenizer SFT experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.train_sft import train_from_config
from src.utils.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tokenizer_config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. training.learning_rate=5e-5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.tokenizer_config, overrides=args.overrides)
    run_dir = train_from_config(config)
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
