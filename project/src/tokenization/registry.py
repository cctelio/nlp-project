"""Tokenizer strategy registry and Hugging Face integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tokenization.base import (
    AtomwiseSmilesRepresentation,
    DefaultSmilesRepresentation,
    SmilesPESmilesRepresentation,
    SmilesRepresentation,
    TokenizerBuildResult,
    tokenized_length_summary,
)


def _require_transformers():
    try:
        from transformers import AddedToken, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for tokenizer construction.") from exc
    return AddedToken, AutoTokenizer


def build_representation(strategy_config: dict[str, Any], train_smiles: list[str]):
    """Construct the chemical SMILES representation from config."""
    strategy = strategy_config.get("strategy", "default")
    strict = bool(strategy_config.get("strict_regex", False))
    if strategy == "default":
        return DefaultSmilesRepresentation()
    if strategy == "atomwise":
        return AtomwiseSmilesRepresentation(strict=strict)
    if strategy == "smilespe":
        return SmilesPESmilesRepresentation.train(
            train_smiles,
            vocab_size=int(strategy_config.get("vocab_size", 500)),
            min_frequency=int(strategy_config.get("min_frequency", 2)),
            strict=strict,
        )
    raise ValueError(f"Unknown tokenizer strategy: {strategy}")


def build_tokenizer(
    model_id: str,
    strategy_config: dict[str, Any],
    train_smiles: list[str],
    split_smiles: dict[str, list[str]] | None = None,
    tokenizer_revision: str | None = None,
) -> TokenizerBuildResult:
    """Load the base tokenizer, add chemical tokens, and compute tokenizer metrics."""
    representation = build_representation(strategy_config, train_smiles=train_smiles)
    return build_tokenizer_for_representation(
        model_id=model_id,
        representation=representation,
        train_smiles=train_smiles,
        split_smiles=split_smiles,
        tokenizer_revision=tokenizer_revision,
    )


def build_tokenizer_for_representation(
    model_id: str,
    representation: SmilesRepresentation,
    train_smiles: list[str],
    split_smiles: dict[str, list[str]] | None = None,
    tokenizer_revision: str | None = None,
) -> TokenizerBuildResult:
    """Load the base tokenizer, add tokens from an existing representation, and compute metrics."""
    AddedToken, AutoTokenizer = _require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    initial_vocab_size = len(tokenizer)
    candidate_tokens = representation.added_tokens(train_smiles)
    added_token_objects = [
        AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for token in candidate_tokens
        if token
    ]
    add_tokens_return_count = tokenizer.add_tokens(added_token_objects)
    final_vocab_size = len(tokenizer)
    actual_added_count = final_vocab_size - initial_vocab_size

    metrics: dict[str, Any] = {
        "tokenizer_strategy": representation.strategy,
        "base_vocab_size": initial_vocab_size,
        "candidate_added_token_count": len(candidate_tokens),
        "added_token_count": int(actual_added_count),
        "hf_add_tokens_return_count": int(add_tokens_return_count),
        "final_vocab_size": final_vocab_size,
        "vocab_expansion_fraction": float(actual_added_count / initial_vocab_size) if initial_vocab_size else 0.0,
    }

    for split_name, smiles_values in (split_smiles or {"train": train_smiles}).items():
        summary = tokenized_length_summary(tokenizer, representation, smiles_values)
        for key, value in summary.items():
            metrics[f"{split_name}_smiles_token_length_{key}"] = value

    return TokenizerBuildResult(
        tokenizer=tokenizer,
        representation=representation,
        metrics=metrics,
        added_tokens=candidate_tokens,
    )


def save_tokenizer_artifacts(result: TokenizerBuildResult, output_dir: str | Path) -> None:
    """Save tokenizer and chemical representation artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.tokenizer.save_pretrained(output_dir / "tokenizer")
    result.representation.save(output_dir / "smiles_representation.json")
