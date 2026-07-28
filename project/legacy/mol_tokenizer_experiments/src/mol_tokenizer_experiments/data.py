from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download


DATASET_ID = "zjunlp/Mol-Instructions"
MOLECULE_ZIP_PATH = "data/Molecule-oriented_Instructions.zip"
TASK_JSON_NAME = "Molecule-oriented_Instructions/description_guided_molecule_design.json"
DEFAULT_SEED = 42


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _looks_like_smiles(value: str) -> bool:
    if not value:
        return False
    if len(value.split()) > 1:
        return False
    return bool(re.search(r"[A-Za-z\[\]\(\)=#@+\-\\/0-9]", value))


def normalize_molinstructions_row(row: dict) -> dict:
    instruction = _clean_text(row.get("instruction"))
    input_text = _clean_text(row.get("input"))
    output = _clean_text(row.get("output"))
    if input_text and input_text.lower() not in {"none", "nan", "null"}:
        prompt = f"{instruction}\n{input_text}" if instruction else input_text
    else:
        prompt = instruction
    return {
        "instruction": prompt,
        "target_smiles": output,
        "metadata": _clean_text(row.get("metadata")),
    }


def load_description_guided_dataset(
    seed: int = DEFAULT_SEED,
    test_size: float = 0.1,
) -> DatasetDict:
    zip_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=MOLECULE_ZIP_PATH,
        repo_type="dataset",
    )
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(TASK_JSON_NAME) as handle:
            df = pd.read_json(handle)
    raw = Dataset.from_pandas(df, preserve_index=False)
    normalized = raw.map(
        normalize_molinstructions_row,
        remove_columns=raw.column_names,
        desc="Normalizing Mol-Instructions rows",
    )
    normalized = normalized.filter(
        lambda row: bool(row["instruction"]) and _looks_like_smiles(row["target_smiles"]),
        desc="Filtering rows with valid-looking prompts and SMILES strings",
    )
    split = normalized.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def limit_dataset_samples(
    dataset: DatasetDict,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DatasetDict:
    train = dataset["train"]
    test = dataset["test"]
    if max_train_samples is not None:
        train = train.select(range(min(max_train_samples, len(train))))
    if max_eval_samples is not None:
        test = test.select(range(min(max_eval_samples, len(test))))
    return DatasetDict({"train": train, "test": test})


def write_prepared_dataset(dataset: DatasetDict, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name, split_dataset in dataset.items():
        path = output_dir / f"{split_name}.csv"
        split_dataset.to_pandas().to_csv(path, index=False)
        paths[split_name] = str(path)
    return paths


def load_prepared_dataset(data_dir: str | Path) -> DatasetDict:
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing prepared train/test CSVs under {data_dir}")
    return DatasetDict(
        {
            "train": Dataset.from_pandas(pd.read_csv(train_path), preserve_index=False),
            "test": Dataset.from_pandas(pd.read_csv(test_path), preserve_index=False),
        }
    )


def get_or_prepare_dataset(
    processed_dir: str | Path,
    seed: int = DEFAULT_SEED,
    test_size: float = 0.1,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    force_prepare: bool = False,
) -> DatasetDict:
    processed_dir = Path(processed_dir)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if train_path.exists() and test_path.exists() and not force_prepare:
        dataset = load_prepared_dataset(processed_dir)
        return limit_dataset_samples(dataset, max_train_samples, max_eval_samples)
    dataset = load_description_guided_dataset(
        seed=seed,
        test_size=test_size,
    )
    write_prepared_dataset(dataset, processed_dir)
    return limit_dataset_samples(dataset, max_train_samples, max_eval_samples)


def to_sft_messages(batch, tokenizer_strategy):
    messages = []
    for instruction, target_smiles in zip(batch["instruction"], batch["target_smiles"]):
        messages.append(
            [
                {"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": tokenizer_strategy.encode_smiles_text(str(target_smiles))},
            ]
        )
    return {"messages": messages}
