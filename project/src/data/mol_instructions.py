"""Mol-Instructions loading and description-guided molecule design filtering."""

from __future__ import annotations

import csv
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.data.preprocessing import normalize_text, row_id, save_splits, split_rows
from src.utils.reproducibility import stable_int_seed

DEFAULT_DATASET_NAME = "zjunlp/Mol-Instructions"
DEFAULT_ARCHIVE = "data/Molecule-oriented_Instructions.zip"

TASK_HINTS = (
    "description-guided molecule design",
    "description guided molecule design",
    "description_guided_molecule_design",
    "description-guided_molecule_design",
    "description guided molecule",
)

INSTRUCTION_FIELDS = ("instruction", "prompt", "input", "question", "description")
RESPONSE_FIELDS = ("output", "response", "answer", "completion", "target", "smiles", "canonical_smiles")
TASK_FIELDS = ("task", "task_name", "category", "subset", "subtask", "source", "source_file", "file")
SPLIT_FIELDS = ("split", "set", "partition")


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for Mol-Instructions preprocessing.") from exc
    return pd


def download_mol_instructions_archive(
    dataset_name: str = DEFAULT_DATASET_NAME,
    archive_filename: str = DEFAULT_ARCHIVE,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download the molecule-oriented Mol-Instructions archive from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download Mol-Instructions.") from exc

    return Path(
        hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=archive_filename,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet"}:
            yield file_path


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "examples", "instances"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_parquet_records(path: Path) -> list[dict[str, Any]]:
    pd = _require_pandas()
    return pd.read_parquet(path).to_dict(orient="records")


def read_records_from_path(path: str | Path, extract_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Read records from a directory, table file, or zip archive."""
    path = Path(path)
    source_root = path
    if path.suffix.lower() == ".zip":
        target_dir = Path(extract_dir) if extract_dir else path.with_suffix("")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target_dir)
        source_root = target_dir

    rows: list[dict[str, Any]] = []
    for file_path in _iter_files(source_root):
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            file_rows = _read_json_records(file_path)
        elif suffix == ".jsonl":
            file_rows = _read_jsonl_records(file_path)
        elif suffix == ".csv":
            file_rows = _read_csv_records(file_path)
        elif suffix == ".parquet":
            file_rows = _read_parquet_records(file_path)
        else:
            file_rows = []
        for row in file_rows:
            row = dict(row)
            row.setdefault("source_file", str(file_path))
            rows.append(row)
    return rows


def _task_haystack(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    values = [normalize_text(row.get(field)) for field in TASK_FIELDS if row.get(field) is not None]
    values.extend(normalize_text(metadata.get(field)) for field in TASK_FIELDS if metadata.get(field) is not None)
    return " ".join(values).lower().replace("-", " ").replace("_", " ")


def is_description_guided_design(row: dict[str, Any], task_filter: str | None = None) -> bool:
    """Return whether a raw row belongs to the description-guided molecule design subset."""
    haystack = _task_haystack(row)
    if task_filter:
        return task_filter.lower().replace("-", " ").replace("_", " ") in haystack
    if any(hint.replace("-", " ").replace("_", " ") in haystack for hint in TASK_HINTS):
        return True
    return "description" in haystack and "molecule" in haystack and "design" in haystack


def _first_non_empty(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = normalize_text(row.get(field))
        if value:
            return value
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for field in fields:
        value = normalize_text(metadata.get(field))
        if value:
            return value
    return ""


def _canonical_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"validation", "valid", "val", "dev", "eval"}:
        return "validation"
    if normalized in {"test", "testing"}:
        return "test"
    return ""


def _looks_like_selfies(value: str) -> bool:
    return value.startswith("[") and "]" in value and value.count("[") >= 2


def _decode_selfies(value: str) -> str:
    try:
        import selfies as sf
    except ImportError as exc:
        raise ImportError(
            "Mol-Instructions description-guided molecule design outputs are SELFIES. "
            "Install `selfies` or use a preprocessed file with decoded SMILES."
        ) from exc
    return sf.decoder(value)


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize a SMILES string with RDKit."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:
        raise ImportError("RDKit is required for canonical SMILES preprocessing.") from exc

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def normalize_molecular_output(value: str, output_format: str = "auto", canonicalize: bool = False) -> tuple[str, str]:
    """Convert a raw Mol-Instructions target into SMILES."""
    raw_value = normalize_text(value)
    if not raw_value:
        return "", ""

    detected_format = output_format
    if output_format == "auto":
        detected_format = "selfies" if _looks_like_selfies(raw_value) else "smiles"

    if detected_format == "selfies":
        smiles = _decode_selfies(raw_value)
    elif detected_format in {"smiles", "raw_smiles"}:
        smiles = raw_value
    else:
        raise ValueError(f"Unsupported molecular output format: {output_format}")

    smiles = canonicalize_smiles(smiles) if canonicalize else smiles
    return smiles, detected_format


def normalize_mol_instruction_record(
    row: dict[str, Any],
    *,
    output_format: str = "auto",
    canonicalize: bool = False,
) -> dict[str, Any] | None:
    """Normalize a Mol-Instructions row into instruction and target SMILES fields."""
    instruction = _first_non_empty(row, INSTRUCTION_FIELDS)
    raw_output = _first_non_empty(row, RESPONSE_FIELDS)

    # Common instruction datasets use both `instruction` and `input`; keep both if they carry distinct content.
    base_instruction = normalize_text(row.get("instruction"))
    input_text = normalize_text(row.get("input"))
    if base_instruction and input_text and input_text not in base_instruction:
        instruction = f"{base_instruction}\n{input_text}"

    if not instruction or not raw_output:
        return None
    target_smiles, detected_format = normalize_molecular_output(
        raw_output,
        output_format=output_format,
        canonicalize=canonicalize,
    )
    if not target_smiles:
        return None

    source = normalize_text(row.get("source_file") or row.get("source") or "")
    normalized = {
        "id": row_id(instruction, target_smiles, source),
        "instruction": instruction,
        "target_smiles": target_smiles,
        "raw_output": raw_output,
        "raw_output_format": detected_format,
        "source": source,
    }
    split = _canonical_split(_first_non_empty(row, SPLIT_FIELDS))
    if split:
        normalized["official_split"] = split
    return normalized


def _split_official_train_test(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    official_train = [row for row in rows if row.get("official_split") == "train"]
    official_test = [row for row in rows if row.get("official_split") == "test"]
    import random

    shuffled_train = list(official_train)
    random.Random(stable_int_seed(seed, "official-train-validation-split")).shuffle(shuffled_train)
    validation_count = max(1, int(len(shuffled_train) * validation_fraction)) if len(shuffled_train) > 1 else 0
    validation_rows = shuffled_train[:validation_count]
    train_rows = shuffled_train[validation_count:]
    split_map = {"train": train_rows, "validation": validation_rows, "test": official_test}
    for split_name, split_rows_ in split_map.items():
        for row in split_rows_:
            row["split"] = split_name
    return split_map


def prepare_description_guided_splits(
    output_dir: str | Path,
    raw_path: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    archive_filename: str = DEFAULT_ARCHIVE,
    cache_dir: str | Path | None = None,
    task_filter: str | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
    output_format: str = "auto",
    canonicalize: bool = True,
) -> dict[str, str]:
    """Prepare deterministic JSONL splits for description-guided molecule design."""
    if raw_path is None:
        raw_path = download_mol_instructions_archive(dataset_name, archive_filename, cache_dir)
    raw_path = Path(raw_path)
    extract_dir = Path(cache_dir) / "mol_instructions_extracted" if cache_dir else None
    raw_rows = read_records_from_path(raw_path, extract_dir=extract_dir)
    filtered_rows = [row for row in raw_rows if is_description_guided_design(row, task_filter=task_filter)]
    normalized_rows = [
        row
        for row in (
            normalize_mol_instruction_record(row, output_format=output_format, canonicalize=canonicalize)
            for row in filtered_rows
        )
        if row is not None
    ]

    deduped: dict[str, dict[str, Any]] = {}
    for row in normalized_rows:
        deduped[row["id"]] = row
    rows = list(deduped.values())
    if max_samples:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(
            "No description-guided molecule design examples were found. "
            "Set data.raw_path to the Mol-Instructions archive/directory and adjust data.task_filter if needed."
        )

    official_split_values = {row.get("official_split") for row in rows}
    if {"train", "validation", "test"}.issubset(official_split_values):
        splits = {
            "train": [row for row in rows if row.get("official_split") == "train"],
            "validation": [row for row in rows if row.get("official_split") == "validation"],
            "test": [row for row in rows if row.get("official_split") == "test"],
        }
        for split_name, split_rows_ in splits.items():
            for row in split_rows_:
                row["split"] = split_name
        split_policy = "official_train_validation_test"
    elif {"train", "test"}.issubset(official_split_values):
        splits = _split_official_train_test(rows, validation_fraction=validation_fraction, seed=seed)
        split_policy = "official_train_test_with_validation_from_train"
    else:
        splits = split_rows(rows, train_fraction=train_fraction, validation_fraction=validation_fraction, seed=seed)
        split_policy = "deterministic_fractional_split"
    manifest = {
        "dataset_name": dataset_name,
        "archive_filename": archive_filename,
        "raw_path": str(raw_path),
        "task_filter": task_filter or "description-guided molecule design",
        "seed": seed,
        "total_raw_records": len(raw_rows),
        "filtered_records": len(filtered_rows),
        "normalized_records": len(normalized_rows),
        "deduplicated_records": len(rows),
        "split_policy": split_policy,
        "output_format": output_format,
        "canonicalize_smiles": canonicalize,
    }
    return save_splits(splits, output_dir, manifest)
