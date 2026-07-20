"""Precompute SFT-ready datasets for TRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.preprocessing import PROMPT_TEMPLATE, format_sft_text
from src.tokenization.registry import build_representation
from src.utils.logging import read_rows, write_csv, write_json, write_jsonl


def make_sft_row(row: dict[str, Any], representation: Any, prompt_template: str = PROMPT_TEMPLATE) -> dict[str, Any]:
    """Create one SFT-ready row with phenoVLM-style messages and plain text."""
    instruction = str(row["instruction"])
    target_smiles = str(row["target_smiles"])
    response_smiles = representation.encode_smiles(target_smiles)
    messages = [
        {"role": "user", "content": instruction.strip()},
        {"role": "assistant", "content": response_smiles},
    ]
    return {
        **row,
        "response_smiles": response_smiles,
        "messages": messages,
        "text": format_sft_text(instruction, response=response_smiles, eos_token="", template=prompt_template),
    }


def precompute_sft_splits(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path | None,
    output_dir: str | Path,
    tokenizer_config: dict[str, Any],
    prompt_template: str = PROMPT_TEMPLATE,
) -> dict[str, str]:
    """Write SFT-ready JSONL and CSV splits for one tokenizer strategy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_rows(train_path)
    validation_rows = read_rows(validation_path)
    test_rows = read_rows(test_path) if test_path else []
    train_smiles = [str(row["target_smiles"]) for row in train_rows]
    representation = build_representation(tokenizer_config, train_smiles=train_smiles)

    split_map = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }
    paths: dict[str, str] = {}
    for split_name, rows in split_map.items():
        sft_rows = [make_sft_row(row, representation, prompt_template=prompt_template) for row in rows]
        jsonl_path = output_dir / f"{split_name}.jsonl"
        csv_path = output_dir / f"{split_name}.csv"
        write_jsonl(jsonl_path, sft_rows)
        write_csv(csv_path, sft_rows)
        paths[f"{split_name}_path"] = str(jsonl_path)
        paths[f"{split_name}_csv_path"] = str(csv_path)

    representation_path = output_dir / "smiles_representation.json"
    representation.save(representation_path)
    manifest = {
        "tokenizer_strategy": representation.strategy,
        "tokenizer_config": tokenizer_config,
        "representation_path": str(representation_path),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_count": len(test_rows),
        "split_paths": paths,
    }
    write_json(output_dir / "manifest.json", manifest)
    paths["representation_path"] = str(representation_path)
    paths["manifest_path"] = str(output_dir / "manifest.json")
    return paths
