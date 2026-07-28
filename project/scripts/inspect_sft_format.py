#!/usr/bin/env python
"""Inspect SFT rows and TRL config without starting training."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import PROMPT_TEMPLATE
from src.training.train_sft import _make_sft_config, _prepare_sft_dataset
from src.tokenization.registry import build_tokenizer
from src.utils.config import load_experiment_config
from src.utils.logging import read_rows
from src.utils.paths import resolve_path


def _short(value: Any, limit: int = 500) -> str:
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else f"{text[:limit]}..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tokenizer_config", default=None)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.tokenizer_config, overrides=args.overrides)

    try:
        from datasets import Dataset
        from transformers import AutoTokenizer  # noqa: F401
        from trl import SFTConfig
        import trl
    except ImportError as exc:
        raise ImportError("Inspection requires datasets, transformers, and trl in the active environment.") from exc

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    tokenizer_config = config.get("tokenizer", {})
    training_config = config.get("training", {})

    train_path = resolve_path(data_config.get("train_path", "results/data/train.jsonl"))
    rows = read_rows(train_path)
    preview_rows = rows[: max(1, args.examples)]
    train_smiles = [str(row["target_smiles"]) for row in rows]
    tokenizer_result = build_tokenizer(
        model_id=model_config.get("model_id", "HuggingFaceTB/SmolLM-135M"),
        strategy_config=tokenizer_config,
        train_smiles=train_smiles,
        split_smiles={"train": train_smiles},
        tokenizer_revision=model_config.get("revision"),
    )

    prompt_template = data_config.get("prompt_template") or PROMPT_TEMPLATE
    sft_format = data_config.get("sft_format", "text")
    use_chat_template = bool(data_config.get("use_chat_template", False))
    dataset = _prepare_sft_dataset(
        Dataset,
        preview_rows,
        tokenizer_result.representation,
        tokenizer_result.tokenizer,
        prompt_template,
        use_chat_template=use_chat_template,
        sft_format=sft_format,
    )

    with TemporaryDirectory() as temp_dir:
        sft_config = _make_sft_config(SFTConfig, Path(temp_dir), config, report_to="none")

    print(f"TRL_VERSION={trl.__version__}")
    print(f"SFTCONFIG_HAS_COMPLETION_ONLY_LOSS={'completion_only_loss' in inspect.signature(SFTConfig.__init__).parameters}")
    print(f"SFT_FORMAT={sft_format}")
    print(f"USE_CHAT_TEMPLATE={use_chat_template}")
    print(f"TRAIN_PATH={train_path}")
    print(f"TRAIN_ROWS={len(rows)}")
    print(f"TOKENIZER_STRATEGY={tokenizer_result.representation.strategy}")
    print(f"MODEL_ID={model_config.get('model_id')}")
    for key in ("completion_only_loss", "loss_type", "packing", "packing_strategy", "padding_free", "max_length"):
        print(f"SFTCONFIG_{key.upper()}={getattr(sft_config, key, None)}")
    print(f"TRAINING_COMPLETION_ONLY_LOSS={training_config.get('completion_only_loss')}")

    for index, item in enumerate(dataset):
        print(f"\nEXAMPLE={index}")
        print(f"ID={item.get('id', '')}")
        print(f"PROMPT={_short(item.get('prompt', ''))}")
        print(f"COMPLETION={_short(item.get('completion', ''))}")
        print(f"TEXT={_short(item.get('text', ''))}")
        prompt_len = len(tokenizer_result.tokenizer.encode(str(item.get("prompt", "")), add_special_tokens=False))
        completion_len = len(tokenizer_result.tokenizer.encode(str(item.get("completion", "")), add_special_tokens=False))
        text_len = len(tokenizer_result.tokenizer.encode(str(item.get("text", "")), add_special_tokens=False))
        print(f"PROMPT_TOKENS={prompt_len}")
        print(f"COMPLETION_TOKENS={completion_len}")
        print(f"TEXT_TOKENS={text_len}")


if __name__ == "__main__":
    main()
