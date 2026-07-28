from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


RDLogger.DisableLog("rdApp.*")
RADIUS = 2
NBITS = 2048


def canonicalize_smiles(smiles: str) -> tuple[bool, str, object | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "", None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "", None
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS)
        return True, canon, fp
    except Exception:
        return False, "", None


def tanimoto_similarity(target_fp, generated_fp) -> float:
    if target_fp is None or generated_fp is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(target_fp, generated_fp))


def compute_generation_metrics(records: list[dict]) -> tuple[dict, pd.DataFrame]:
    rows = []
    for record in records:
        target_valid, target_canon, target_fp = canonicalize_smiles(record["target_smiles"])
        pred_valid, pred_canon, pred_fp = canonicalize_smiles(record["predicted_smiles"])
        rows.append(
            {
                **record,
                "target_valid": target_valid,
                "target_canonical_smiles": target_canon,
                "predicted_valid": pred_valid,
                "predicted_canonical_smiles": pred_canon,
                "raw_exact_match": str(record["predicted_smiles"]).strip() == str(record["target_smiles"]).strip(),
                "canonical_exact_match": bool(target_valid and pred_valid and target_canon == pred_canon),
                "tanimoto_similarity": tanimoto_similarity(target_fp, pred_fp),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "eval_count": 0,
            "raw_exact_match": 0.0,
            "canonical_exact_match": 0.0,
            "validity": 0.0,
            "mean_tanimoto_similarity": 0.0,
            "median_tanimoto_similarity": 0.0,
            "unique_valid_predictions": 0,
            "top1_prediction_fraction": 0.0,
        }, df
    valid_canon = [value for value in df.loc[df["predicted_valid"], "predicted_canonical_smiles"] if value]
    counts = Counter(valid_canon)
    metrics = {
        "eval_count": int(len(df)),
        "raw_exact_match": float(df["raw_exact_match"].mean()),
        "canonical_exact_match": float(df["canonical_exact_match"].mean()),
        "validity": float(df["predicted_valid"].mean()),
        "mean_tanimoto_similarity": float(df["tanimoto_similarity"].mean()),
        "median_tanimoto_similarity": float(np.median(df["tanimoto_similarity"])),
        "unique_valid_predictions": int(len(counts)),
        "top1_prediction_fraction": float(max(counts.values()) / len(df)) if counts else 0.0,
    }
    return metrics, df
