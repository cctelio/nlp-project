"""Classical regex-based atom-wise SMILES tokenization."""

from __future__ import annotations

import re
from collections import Counter

# Schwaller/Molecular Transformer style SMILES regex, also used by common chemistry tokenizers.
SMILES_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
SMILES_REGEX = re.compile(SMILES_REGEX_PATTERN)


def regex_tokenize_smiles(smiles: str, strict: bool = True) -> list[str]:
    """Tokenize a SMILES string with the classical atom-wise regex."""
    if not smiles:
        return []
    tokens: list[str] = []
    cursor = 0
    for match in SMILES_REGEX.finditer(smiles):
        if match.start() > cursor:
            if strict:
                break
            tokens.extend(smiles[cursor : match.start()])
        tokens.append(match.group(0))
        cursor = match.end()
    if cursor < len(smiles) and not strict:
        tokens.extend(smiles[cursor:])
    if strict and "".join(tokens) != smiles:
        raise ValueError(f"SMILES regex failed to cover the full string: {smiles!r} -> {tokens!r}")
    return tokens


def detokenize_smiles_tokens(tokens: list[str]) -> str:
    """Convert atom-wise or SPE tokens back to a raw SMILES string."""
    return "".join(token for token in tokens if token)


def token_frequencies(smiles_values: list[str], strict: bool = False) -> Counter[str]:
    """Count regex tokens in a SMILES corpus."""
    counts: Counter[str] = Counter()
    for smiles in smiles_values:
        try:
            counts.update(regex_tokenize_smiles(smiles, strict=strict))
        except ValueError:
            if strict:
                raise
    return counts


def ordered_tokens_by_frequency(counts: Counter[str]) -> list[str]:
    """Sort tokens by descending frequency with lexical tie-breaking."""
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
