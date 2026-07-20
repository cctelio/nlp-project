"""Shared tokenizer strategy abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tokenization.atomwise import (
    SMILES_REGEX_PATTERN,
    detokenize_smiles_tokens,
    ordered_tokens_by_frequency,
    regex_tokenize_smiles,
    token_frequencies,
)
from src.tokenization.smilespe import MergeRule, apply_merges, learn_smilespe_merges
from src.utils.logging import read_json, write_json


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text and text.strip() else ""


class SmilesRepresentation(ABC):
    """A training-time representation for target SMILES strings."""

    strategy: str

    @abstractmethod
    def encode_smiles(self, smiles: str) -> str:
        """Convert raw SMILES into the representation used in SFT responses."""

    @abstractmethod
    def decode_generation(self, text: str) -> str:
        """Convert generated text back to a raw SMILES candidate."""

    @abstractmethod
    def representation_tokens(self, smiles: str) -> list[str]:
        """Return representation-level tokens for analysis."""

    @abstractmethod
    def added_tokens(self, train_smiles: list[str]) -> list[str]:
        """Return tokens that should be added to the base HF tokenizer."""

    @abstractmethod
    def to_artifact(self) -> dict[str, Any]:
        """Serialize the representation."""

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_artifact())


class DefaultSmilesRepresentation(SmilesRepresentation):
    """Use raw SMILES with the unmodified SmolLM tokenizer."""

    strategy = "default"

    def encode_smiles(self, smiles: str) -> str:
        return smiles.strip()

    def decode_generation(self, text: str) -> str:
        return _first_line(text)

    def representation_tokens(self, smiles: str) -> list[str]:
        return [smiles.strip()] if smiles.strip() else []

    def added_tokens(self, train_smiles: list[str]) -> list[str]:
        return []

    def to_artifact(self) -> dict[str, Any]:
        return {"strategy": self.strategy}


class AtomwiseSmilesRepresentation(SmilesRepresentation):
    """Whitespace-separated classical regex tokens."""

    strategy = "atomwise"

    def __init__(self, strict: bool = False, regex_pattern: str = SMILES_REGEX_PATTERN) -> None:
        self.strict = strict
        self.regex_pattern = regex_pattern

    def encode_smiles(self, smiles: str) -> str:
        return " ".join(regex_tokenize_smiles(smiles, strict=self.strict))

    def decode_generation(self, text: str) -> str:
        candidate = _first_line(text)
        return detokenize_smiles_tokens(candidate.split())

    def representation_tokens(self, smiles: str) -> list[str]:
        return regex_tokenize_smiles(smiles, strict=self.strict)

    def added_tokens(self, train_smiles: list[str]) -> list[str]:
        return ordered_tokens_by_frequency(token_frequencies(train_smiles, strict=self.strict))

    def to_artifact(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "strict": self.strict, "regex_pattern": self.regex_pattern}


class SmilesPESmilesRepresentation(SmilesRepresentation):
    """SMILES Pair Encoding representation with learned adjacent token merges."""

    strategy = "smilespe"

    def __init__(self, merges: list[MergeRule], vocabulary: list[str], strict: bool = False) -> None:
        self.merges = merges
        self.vocabulary = vocabulary
        self.strict = strict

    @classmethod
    def train(
        cls,
        train_smiles: list[str],
        vocab_size: int,
        min_frequency: int = 2,
        strict: bool = False,
    ) -> SmilesPESmilesRepresentation:
        merges, vocabulary = learn_smilespe_merges(
            train_smiles,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            strict=strict,
        )
        return cls(merges=merges, vocabulary=vocabulary, strict=strict)

    def encode_smiles(self, smiles: str) -> str:
        tokens = regex_tokenize_smiles(smiles, strict=self.strict)
        return " ".join(apply_merges(tokens, self.merges))

    def decode_generation(self, text: str) -> str:
        candidate = _first_line(text)
        return detokenize_smiles_tokens(candidate.split())

    def representation_tokens(self, smiles: str) -> list[str]:
        return apply_merges(regex_tokenize_smiles(smiles, strict=self.strict), self.merges)

    def added_tokens(self, train_smiles: list[str]) -> list[str]:
        return sorted(self.vocabulary, key=lambda token: (-len(token), token))

    def to_artifact(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "strict": self.strict,
            "vocabulary": self.vocabulary,
            "merges": [merge.to_dict() for merge in self.merges],
        }

    @classmethod
    def from_artifact(cls, payload: dict[str, Any]) -> SmilesPESmilesRepresentation:
        merges = [MergeRule.from_dict(item) for item in payload.get("merges", [])]
        vocabulary = [str(item) for item in payload.get("vocabulary", [])]
        return cls(merges=merges, vocabulary=vocabulary, strict=bool(payload.get("strict", False)))


@dataclass
class TokenizerBuildResult:
    """A built Hugging Face tokenizer plus chemical representation metadata."""

    tokenizer: Any
    representation: SmilesRepresentation
    metrics: dict[str, Any]
    added_tokens: list[str]


def load_representation(path: str | Path) -> SmilesRepresentation:
    """Load a saved representation artifact."""
    payload = read_json(path)
    strategy = payload.get("strategy")
    if strategy == "default":
        return DefaultSmilesRepresentation()
    if strategy == "atomwise":
        return AtomwiseSmilesRepresentation(
            strict=bool(payload.get("strict", False)),
            regex_pattern=str(payload.get("regex_pattern", SMILES_REGEX_PATTERN)),
        )
    if strategy == "smilespe":
        return SmilesPESmilesRepresentation.from_artifact(payload)
    raise ValueError(f"Unknown tokenizer representation strategy: {strategy}")


def tokenized_length_summary(
    tokenizer: Any,
    representation: SmilesRepresentation,
    smiles_values: list[str],
) -> dict[str, float]:
    """Summarize HF tokenized lengths of represented SMILES strings."""
    lengths = [
        len(tokenizer.encode(representation.encode_smiles(smiles), add_special_tokens=False))
        for smiles in smiles_values
        if smiles
    ]
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0, "max": 0.0}
    values = sorted(float(value) for value in lengths)

    def quantile(q: float) -> float:
        if len(values) == 1:
            return values[0]
        position = q * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "mean": float(sum(values) / len(values)),
        "median": float(quantile(0.50)),
        "q25": float(quantile(0.25)),
        "q75": float(quantile(0.75)),
        "max": float(max(values)),
    }
