import pytest

from src.evaluation.metrics import evaluate_generation_records

rdkit = pytest.importorskip("rdkit")


def test_generation_metrics_exact_valid_tanimoto():
    records = [{"target_smiles": "CCO", "generated_smiles": "CCO"}]
    metrics, enriched = evaluate_generation_records(records)
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["canonical_exact_match_accuracy"] == 1.0
    assert metrics["validity"] == 1.0
    assert metrics["tanimoto_similarity_mean"] == 1.0
    assert enriched[0]["valid"]
