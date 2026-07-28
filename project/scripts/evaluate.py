#!/usr/bin/env python
"""Evaluate a trained run on validation or test data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import PROMPT_TEMPLATE
from src.evaluation.evaluate import evaluate_rows
from src.tokenization.base import load_representation
from src.utils.config import load_yaml
from src.utils.logging import read_jsonl
from src.utils.paths import resolve_path
from src.utils.reproducibility import stable_int_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_examples", type=int, default=None)
    return parser.parse_args()


def _resolve_dtype(torch, dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return None


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    data = config.get("data", {})
    model_config = config.get("model", {})
    split_path = resolve_path(data.get(f"{args.split}_path"))
    rows = read_jsonl(split_path)
    if args.max_examples:
        rows = rows[: args.max_examples]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Evaluation requires torch and transformers.") from exc

    final_model_dir = run_dir / "final_model"
    tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _resolve_dtype(torch, model_config.get("torch_dtype", "bfloat16"))
    kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(final_model_dir, **kwargs)
    if torch.cuda.is_available() is False:
        model.to("cpu")

    representation = load_representation(final_model_dir / "smiles_representation.json")
    generation = dict(config.get("generation", {}))
    generation["seed"] = stable_int_seed(int(config.get("seed", 42)), f"{args.split}-generation")
    metrics = evaluate_rows(
        model=model,
        tokenizer=tokenizer,
        representation=representation,
        rows=rows,
        split_name=args.split,
        output_dir=run_dir / "evaluation",
        generation_config=generation,
        prompt_template=data.get("prompt_template") or PROMPT_TEMPLATE,
        use_chat_template=bool(data.get("use_chat_template", False)),
    )
    print(metrics)


if __name__ == "__main__":
    main()
