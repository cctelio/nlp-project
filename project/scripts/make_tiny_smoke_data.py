#!/usr/bin/env python
"""Create tiny canonical CSV splits for infrastructure smoke tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import row_id
from src.utils.logging import write_csv, write_json


EXAMPLES = [
    ("Design a small alcohol with two carbon atoms.", "CCO"),
    ("Design a simple amine with two carbon atoms.", "CCN"),
    ("Design acetic acid.", "CC(=O)O"),
    ("Design benzene.", "c1ccccc1"),
    ("Design chloroethane.", "CCCl"),
    ("Design fluoroethane.", "CCF"),
    ("Design cyclohexane.", "C1CCCCC1"),
    ("Design acetone.", "CC(=O)C"),
    ("Design ethyl acetate.", "CCOC(=O)C"),
    ("Design pyridine.", "c1ccncc1"),
    ("Design ethanolamine.", "NCCO"),
    ("Design formamide.", "NC=O"),
    ("Design dimethyl ether.", "COC"),
    ("Design propanol.", "CCCO"),
    ("Design propylamine.", "CCCN"),
    ("Design phenol.", "Oc1ccccc1"),
]


def _rows(split: str, examples: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "id": row_id(instruction, smiles, source="tiny-smoke"),
            "instruction": instruction,
            "target_smiles": smiles,
            "split": split,
            "source": "tiny-smoke",
        }
        for instruction, smiles in examples
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="results/data/tiny_smoke")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    split_map = {
        "train": _rows("train", EXAMPLES[:12]),
        "validation": _rows("validation", EXAMPLES[12:14]),
        "test": _rows("test", EXAMPLES[14:]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_map.items():
        write_csv(output_dir / f"{split_name}.csv", rows)
    write_json(
        output_dir / "manifest.json",
        {
            "source": "tiny-smoke",
            "purpose": "Infrastructure-only smoke data; not for metrics or conclusions.",
            "train_count": len(split_map["train"]),
            "validation_count": len(split_map["validation"]),
            "test_count": len(split_map["test"]),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
