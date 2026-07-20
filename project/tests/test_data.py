from src.data.mol_instructions import is_description_guided_design, normalize_mol_instruction_record
from src.data.preprocessing import split_rows
from src.data.sft import make_sft_row
from src.tokenization.base import AtomwiseSmilesRepresentation


def test_description_guided_filter():
    row = {"task": "Description-guided molecule design"}
    assert is_description_guided_design(row)


def test_record_normalization():
    row = {
        "instruction": "Design a molecule with this property.",
        "input": "It should be small.",
        "output": "CCO",
        "source_file": "Description-guided_Molecule_Design.json",
    }
    normalized = normalize_mol_instruction_record(row)
    assert normalized is not None
    assert "small" in normalized["instruction"]
    assert normalized["target_smiles"] == "CCO"


def test_record_normalization_preserves_official_split():
    row = {"instruction": "Design", "output": "CCO", "split": "valid"}
    normalized = normalize_mol_instruction_record(row)
    assert normalized is not None
    assert normalized["official_split"] == "validation"


def test_split_rows_is_deterministic():
    rows = [{"id": str(idx), "instruction": "x", "target_smiles": "C"} for idx in range(20)]
    first = split_rows(rows, 0.8, 0.1, seed=42)
    second = split_rows(rows, 0.8, 0.1, seed=42)
    assert [row["id"] for row in first["train"]] == [row["id"] for row in second["train"]]
    assert len(first["train"]) == 16
    assert len(first["validation"]) == 2
    assert len(first["test"]) == 2


def test_make_sft_row_has_messages_and_text():
    row = {"id": "1", "instruction": "Design ethanol.", "target_smiles": "CCO"}
    sft_row = make_sft_row(row, AtomwiseSmilesRepresentation())
    assert sft_row["response_smiles"] == "C C O"
    assert sft_row["messages"][0]["role"] == "user"
    assert sft_row["messages"][1]["content"] == "C C O"
    assert "### Response:" in sft_row["text"]
