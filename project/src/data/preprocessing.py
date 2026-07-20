"""Dataset normalization and split helpers."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from src.utils.logging import write_csv, write_json, write_jsonl
from src.utils.reproducibility import stable_int_seed

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def normalize_text(value: Any) -> str:
    """Convert a field to a compact string."""
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def format_generation_prompt(instruction: str, template: str = PROMPT_TEMPLATE) -> str:
    """Format the prompt used for generation."""
    return template.format(instruction=instruction.strip())


def format_sft_text(instruction: str, response: str, eos_token: str = "", template: str = PROMPT_TEMPLATE) -> str:
    """Format one instruction/response example for causal LM SFT."""
    eos = eos_token or ""
    return f"{format_generation_prompt(instruction, template)}{response.strip()}{eos}"


def row_id(instruction: str, target_smiles: str, source: str = "") -> str:
    """Create a stable example id from core content."""
    digest = hashlib.sha256(f"{instruction}\n{target_smiles}\n{source}".encode()).hexdigest()
    return digest[:16]


def split_rows(
    rows: list[dict[str, Any]],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic train/validation/test splits."""
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Expected train_fraction > 0, validation_fraction >= 0, and train+validation < 1")

    shuffled = list(rows)
    random.Random(stable_int_seed(seed, "mol-instructions-split")).shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    split_map = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    for split_name, split_rows_ in split_map.items():
        for row in split_rows_:
            row["split"] = split_name
    return split_map


def save_splits(
    split_map: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Write canonical JSONL and CSV splits plus a manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split_name, rows in split_map.items():
        jsonl_path = output_dir / f"{split_name}.jsonl"
        csv_path = output_dir / f"{split_name}.csv"
        write_jsonl(jsonl_path, rows)
        write_csv(csv_path, rows)
        paths[split_name] = str(jsonl_path)
        paths[f"{split_name}_csv"] = str(csv_path)
        manifest[f"{split_name}_count"] = len(rows)
    manifest["split_paths"] = paths
    write_json(output_dir / "manifest.json", manifest)
    return paths
