"""Run evaluation for a model and split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.generate import generate_for_rows
from src.evaluation.metrics import evaluate_generation_records
from src.utils.logging import write_csv, write_json


def evaluate_rows(
    model: Any,
    tokenizer: Any,
    representation: Any,
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    output_dir: str | Path,
    generation_config: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    """Generate outputs for rows, compute metrics, and save artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = generate_for_rows(
        model=model,
        tokenizer=tokenizer,
        representation=representation,
        rows=rows,
        batch_size=int(generation_config.get("batch_size", 16)),
        max_new_tokens=int(generation_config.get("max_new_tokens", 128)),
        temperature=float(generation_config.get("temperature", 1.0)),
        top_p=float(generation_config.get("top_p", 0.95)),
        do_sample=bool(generation_config.get("do_sample", False)),
        seed=int(generation_config.get("seed", 42)),
        prompt_template=prompt_template,
    )
    metrics, enriched_records = evaluate_generation_records(records, tokenizer=tokenizer, representation=representation)
    prefixed_metrics = {f"{split_name}/{key}": value for key, value in metrics.items()}
    write_json(output_dir / f"{split_name}_metrics.json", prefixed_metrics)
    write_csv(output_dir / f"{split_name}_generations.csv", enriched_records)
    return prefixed_metrics
