"""SMILES generation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RADIUS = 2
NBITS = 2048


def _require_rdkit():
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:
        raise ImportError("RDKit is required for chemical evaluation metrics.") from exc
    RDLogger.DisableLog("rdApp.*")
    return Chem, DataStructs, AllChem, rdMolStandardize


@dataclass
class ProcessedSmiles:
    """Canonicalized SMILES and fingerprint state."""

    valid: bool
    canonical_smiles: str
    fingerprint: Any


def process_smiles(smiles: str) -> ProcessedSmiles:
    """Parse, standardize, canonicalize, and fingerprint a SMILES string."""
    Chem, _, AllChem, rdMolStandardize = _require_rdkit()
    if not isinstance(smiles, str) or not smiles.strip():
        return ProcessedSmiles(valid=False, canonical_smiles="", fingerprint=None)
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return ProcessedSmiles(valid=False, canonical_smiles="", fingerprint=None)
        standardized = rdMolStandardize.FragmentParent(mol)
        standardized = rdMolStandardize.Uncharger().uncharge(standardized)
        canonical = Chem.MolToSmiles(standardized, canonical=True, isomericSmiles=True)
        fp = AllChem.GetMorganFingerprintAsBitVect(standardized, RADIUS, nBits=NBITS)
        return ProcessedSmiles(valid=True, canonical_smiles=canonical, fingerprint=fp)
    except Exception:
        return ProcessedSmiles(valid=False, canonical_smiles="", fingerprint=None)


def tanimoto_similarity(target: ProcessedSmiles, generated: ProcessedSmiles, invalid_similarity: float = 0.0) -> float:
    """Compute Morgan fingerprint Tanimoto similarity."""
    _, DataStructs, _, _ = _require_rdkit()
    if target.fingerprint is None or generated.fingerprint is None:
        return invalid_similarity
    return float(DataStructs.TanimotoSimilarity(target.fingerprint, generated.fingerprint))


def _quantiles(values: list[float], prefix: str, empty_value: float = 0.0) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": empty_value,
            f"{prefix}_q25": empty_value,
            f"{prefix}_median": empty_value,
            f"{prefix}_q75": empty_value,
        }
    sorted_values = sorted(float(value) for value in values)

    def quantile(q: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = q * (len(sorted_values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = position - lower
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    return {
        f"{prefix}_mean": float(sum(sorted_values) / len(sorted_values)),
        f"{prefix}_q25": float(quantile(0.25)),
        f"{prefix}_median": float(quantile(0.50)),
        f"{prefix}_q75": float(quantile(0.75)),
    }


def _tokenized_length(tokenizer: Any, representation: Any, smiles: str) -> int:
    try:
        represented = representation.encode_smiles(smiles)
        return int(len(tokenizer.encode(represented, add_special_tokens=False)))
    except Exception:
        return 0


def evaluate_generation_records(
    records: list[dict[str, Any]],
    tokenizer: Any | None = None,
    representation: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute exact-match, validity, Tanimoto, and length metrics."""
    enriched: list[dict[str, Any]] = []
    raw_exact_values: list[bool] = []
    canonical_exact_values: list[bool] = []
    validity_values: list[bool] = []
    empty_values: list[bool] = []
    tanimoto_values: list[float] = []
    target_lengths: list[int] = []
    generated_lengths: list[int] = []

    for record in records:
        target_smiles = str(record.get("target_smiles", "")).strip()
        generated_smiles = str(record.get("generated_smiles", "")).strip()
        target_processed = process_smiles(target_smiles)
        generated_processed = process_smiles(generated_smiles)
        tanimoto = tanimoto_similarity(target_processed, generated_processed)
        raw_exact = target_smiles == generated_smiles
        canonical_exact = (
            target_processed.valid
            and generated_processed.valid
            and target_processed.canonical_smiles == generated_processed.canonical_smiles
        )

        if tokenizer is not None and representation is not None:
            target_lengths.append(_tokenized_length(tokenizer, representation, target_smiles))
            generated_lengths.append(_tokenized_length(tokenizer, representation, generated_smiles))

        raw_exact_values.append(raw_exact)
        canonical_exact_values.append(canonical_exact)
        validity_values.append(generated_processed.valid)
        empty_values.append(not bool(generated_smiles))
        tanimoto_values.append(tanimoto)

        enriched_record = dict(record)
        enriched_record.update(
            {
                "target_canonical_smiles": target_processed.canonical_smiles,
                "generated_canonical_smiles": generated_processed.canonical_smiles,
                "raw_exact_match": raw_exact,
                "canonical_exact_match": canonical_exact,
                "valid": generated_processed.valid,
                "tanimoto_similarity": tanimoto,
            }
        )
        enriched.append(enriched_record)

    metrics: dict[str, Any] = {
        "record_count": len(records),
        "exact_match_accuracy": float(sum(raw_exact_values) / len(raw_exact_values)) if raw_exact_values else 0.0,
        "canonical_exact_match_accuracy": float(sum(canonical_exact_values) / len(canonical_exact_values))
        if canonical_exact_values
        else 0.0,
        "validity": float(sum(validity_values) / len(validity_values)) if validity_values else 0.0,
        "invalid_rate": 1.0 - float(sum(validity_values) / len(validity_values)) if validity_values else 0.0,
        "empty_generation_rate": float(sum(empty_values) / len(empty_values)) if empty_values else 0.0,
    }
    metrics.update(_quantiles(tanimoto_values, "tanimoto_similarity"))
    if target_lengths:
        metrics.update(_quantiles([float(value) for value in target_lengths], "target_tokenized_smiles_length"))
    if generated_lengths:
        metrics.update(_quantiles([float(value) for value in generated_lengths], "generated_tokenized_smiles_length"))
    return metrics, enriched
