"""SMILES Pair Encoding style token learning.

This module learns frequent adjacent atom-token merges from training SMILES. The
learned tokens are added to the base Hugging Face tokenizer, but the pretrained
SmolLM tokenizer itself is not replaced.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.tokenization.atomwise import regex_tokenize_smiles


@dataclass(frozen=True)
class MergeRule:
    """One adjacent-token merge rule."""

    left: str
    right: str
    merged: str
    frequency: int

    def to_dict(self) -> dict[str, Any]:
        return {"left": self.left, "right": self.right, "merged": self.merged, "frequency": self.frequency}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MergeRule:
        return cls(
            left=str(payload["left"]),
            right=str(payload["right"]),
            merged=str(payload["merged"]),
            frequency=int(payload.get("frequency", 0)),
        )


def _merge_once(sequence: list[str], left: str, right: str, merged: str) -> list[str]:
    output: list[str] = []
    idx = 0
    while idx < len(sequence):
        if idx < len(sequence) - 1 and sequence[idx] == left and sequence[idx + 1] == right:
            output.append(merged)
            idx += 2
        else:
            output.append(sequence[idx])
            idx += 1
    return output


def apply_merges(tokens: list[str], merges: list[MergeRule]) -> list[str]:
    """Apply learned merges in order."""
    sequence = list(tokens)
    for merge in merges:
        sequence = _merge_once(sequence, merge.left, merge.right, merge.merged)
    return sequence


def _count_pairs(sequences: list[list[str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for sequence in sequences:
        counts.update(zip(sequence, sequence[1:], strict=False))
    return counts


def learn_smilespe_merges(
    smiles_values: list[str],
    vocab_size: int,
    min_frequency: int = 2,
    strict: bool = False,
) -> tuple[list[MergeRule], list[str]]:
    """Learn SMILESPE-style adjacent token merges from training SMILES."""
    sequences: list[list[str]] = []
    for smiles in smiles_values:
        try:
            tokens = regex_tokenize_smiles(smiles, strict=strict)
        except ValueError:
            if strict:
                raise
            continue
        if tokens:
            sequences.append(tokens)

    vocabulary = {token for sequence in sequences for token in sequence}
    merges: list[MergeRule] = []
    target_vocab_size = max(vocab_size, len(vocabulary))

    while len(vocabulary) < target_vocab_size:
        pair_counts = _count_pairs(sequences)
        if not pair_counts:
            break
        selected_pair: tuple[str, str] | None = None
        selected_frequency = 0
        for (left, right), frequency in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
            if frequency < min_frequency:
                break
            if f"{left}{right}" not in vocabulary:
                selected_pair = (left, right)
                selected_frequency = frequency
                break
        if selected_pair is None:
            break
        left, right = selected_pair
        merged = f"{left}{right}"
        merges.append(MergeRule(left=left, right=right, merged=merged, frequency=selected_frequency))
        vocabulary.add(merged)
        sequences = [_merge_once(sequence, left, right, merged) for sequence in sequences]

    return merges, sorted(vocabulary)
