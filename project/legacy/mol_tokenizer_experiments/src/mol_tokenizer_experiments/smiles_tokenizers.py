from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import AddedToken


SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\[\]]+\]|Br?|Cl?|Si?|Se?|Na?|Li?|Mg?|Ca?|Al?|Fe?|Zn?|"
    r"[B-IK-Zb-ik-z]|\%\d{2}|\d|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\*)"
)


def atomwise_tokenize(smiles: str) -> list[str]:
    tokens = SMILES_TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        return list(smiles)
    return tokens


def _pair_counts(corpus: list[list[str]]) -> Counter[tuple[str, str]]:
    counts = Counter()
    for tokens in corpus:
        counts.update(zip(tokens, tokens[1:]))
    return counts


def _merge_pair(tokens: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    output = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            output.append(merged)
            i += 2
        else:
            output.append(tokens[i])
            i += 1
    return output


@dataclass
class TokenizerStrategy:
    name: str

    def fit(self, smiles_values: Iterable[str]):
        return self

    def encode_smiles_tokens(self, smiles: str) -> list[str]:
        return [smiles]

    def encode_smiles_text(self, smiles: str) -> str:
        return smiles

    def decode_generated_text(self, text: str) -> str:
        return text.strip().replace(" ", "")

    def added_tokens(self) -> list[str]:
        return []

    def save(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "tokenizer_strategy.json").open("w") as handle:
            json.dump({"name": self.name}, handle, indent=2)


class DefaultBPEStrategy(TokenizerStrategy):
    def __init__(self):
        super().__init__(name="default_bpe")

    def decode_generated_text(self, text: str) -> str:
        return text.strip()


class AtomwiseStrategy(TokenizerStrategy):
    def __init__(self):
        super().__init__(name="atomwise")
        self._tokens = []

    def fit(self, smiles_values: Iterable[str]):
        self._tokens = sorted({token for smiles in smiles_values for token in atomwise_tokenize(str(smiles))})
        return self

    def encode_smiles_tokens(self, smiles: str) -> list[str]:
        return atomwise_tokenize(smiles)

    def encode_smiles_text(self, smiles: str) -> str:
        return " ".join(self.encode_smiles_tokens(smiles))

    def added_tokens(self) -> list[str]:
        return list(self._tokens)

    def save(self, output_dir: str | Path):
        super().save(output_dir)
        with (Path(output_dir) / "atomwise_tokens.json").open("w") as handle:
            json.dump(self._tokens, handle, indent=2)


class SmilesPEStrategy(TokenizerStrategy):
    def __init__(self, num_merges: int):
        super().__init__(name=f"smilespe_{num_merges}")
        self.num_merges = int(num_merges)
        self.merges: list[tuple[str, str, str]] = []
        self._tokens = []

    def fit(self, smiles_values: Iterable[str]):
        corpus = [atomwise_tokenize(str(smiles)) for smiles in smiles_values]
        for _ in range(self.num_merges):
            counts = _pair_counts(corpus)
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < 2:
                break
            merged = "".join(pair)
            self.merges.append((pair[0], pair[1], merged))
            corpus = [_merge_pair(tokens, pair, merged) for tokens in corpus]
        self._tokens = sorted({token for tokens in corpus for token in tokens})
        return self

    def encode_smiles_tokens(self, smiles: str) -> list[str]:
        tokens = atomwise_tokenize(smiles)
        for left, right, merged in self.merges:
            tokens = _merge_pair(tokens, (left, right), merged)
        return tokens

    def encode_smiles_text(self, smiles: str) -> str:
        return " ".join(self.encode_smiles_tokens(smiles))

    def added_tokens(self) -> list[str]:
        return list(self._tokens)

    def save(self, output_dir: str | Path):
        super().save(output_dir)
        with (Path(output_dir) / "smilespe_merges.json").open("w") as handle:
            json.dump(
                {
                    "num_merges_requested": self.num_merges,
                    "num_merges_learned": len(self.merges),
                    "merges": self.merges,
                    "tokens": self._tokens,
                },
                handle,
                indent=2,
            )


def build_tokenizer_strategy(name: str) -> TokenizerStrategy:
    if name == "default_bpe":
        return DefaultBPEStrategy()
    if name == "atomwise":
        return AtomwiseStrategy()
    if name.startswith("smilespe_"):
        return SmilesPEStrategy(int(name.removeprefix("smilespe_")))
    raise ValueError(f"Unknown tokenizer strategy: {name}")


def add_strategy_tokens_to_tokenizer(tokenizer, strategy: TokenizerStrategy) -> int:
    raw_tokens = strategy.added_tokens()
    if not raw_tokens:
        return 0
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [
        AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for token in raw_tokens
        if token not in existing_vocab
    ]
    if not new_tokens:
        return 0
    return tokenizer.add_tokens(new_tokens)


def average_encoded_smiles_length(tokenizer, strategy: TokenizerStrategy, smiles_values: Iterable[str]) -> float:
    lengths = []
    for smiles in smiles_values:
        text = strategy.encode_smiles_text(str(smiles))
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    return float(sum(lengths) / len(lengths)) if lengths else 0.0


def assert_roundtrip(strategy: TokenizerStrategy, smiles_values: Iterable[str], limit: int = 256):
    if strategy.name == "default_bpe":
        return
    for idx, smiles in enumerate(smiles_values):
        if idx >= limit:
            break
        smiles = str(smiles)
        decoded = strategy.decode_generated_text(strategy.encode_smiles_text(smiles))
        if decoded != smiles:
            raise ValueError(f"{strategy.name} roundtrip failed: {smiles!r} -> {decoded!r}")
