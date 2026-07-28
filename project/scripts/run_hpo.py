#!/usr/bin/env python
"""Run a matched HPO sweep for one tokenizer configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.hpo import run_hpo_from_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tokenizer_config", required=True)
    parser.add_argument("--sweep_config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_hpo_from_files(args.config, args.tokenizer_config, args.sweep_config)
    print(summary.get("best"))


if __name__ == "__main__":
    main()
