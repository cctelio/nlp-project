# Project Architecture

This document describes the main NLP project in `project/`. The assignment notebooks are separate and live in `../assignments/`.

## Goal

The project tests whether chemistry-aware SMILES representations improve a small language model on a molecule generation task.

Input:

```text
Natural-language molecular description
```

Output:

```text
SMILES string
```

Base model:

```text
HuggingFaceTB/SmolLM-135M
```

Dataset:

```text
zjunlp/Mol-Instructions
description-guided molecule design subset
```

## Experiment Variants

All variants keep the original SmolLM tokenizer for natural language. The project only changes how the target SMILES response is represented.

| Strategy | Config | What changes |
| --- | --- | --- |
| Default | `configs/tokenizers/default.yaml` | Raw SMILES target, unchanged tokenizer behavior. |
| Atom-wise | `configs/tokenizers/atomwise.yaml` | SMILES is split into atom/operator tokens like `C C ( = O ) Cl`. |
| SMILESPE | `configs/tokenizers/smilespe.yaml` | Learns SMILES pair-encoding merges from the training SMILES. |

For `atomwise` and `smilespe`, the code adds missing chemistry tokens to the Hugging Face tokenizer and resizes the model embeddings before fine-tuning.

## Directory Map

```text
configs/
  default.yaml              # Shared model, data, training, generation settings
  tokenizers/               # Tokenizer strategy configs
  sweeps/                   # Hyperparameter search spaces

scripts/
  prepare_data.py           # Download/filter/decode/canonicalize Mol-Instructions
  prepare_sft_data.py       # Optional precompute of tokenizer-specific SFT files
  train.py                  # Run one SFT experiment
  evaluate.py               # Evaluate a trained run on validation or test
  run_hpo.py                # Run one hyperparameter sweep
  summarize_hpo.py          # Select/summarize best runs
  run_all_experiments.sh    # Run all tokenizer sweeps locally

src/
  data/                     # Dataset loading, splitting, SFT formatting
  tokenization/             # Default, atom-wise, and SMILESPE representations
  training/                 # TRL SFTTrainer and HPO workflow
  evaluation/               # Generation and chemistry metrics
  utils/                    # Config, logging, paths, reproducibility

tests/                      # Unit tests for data, metrics, tokenization
results/                    # Generated outputs, ignored except `.gitkeep`

legacy/
  mol_tokenizer_experiments/  # Older standalone experiment version
```

## End-to-End Flow

1. `scripts/prepare_data.py`

   Downloads or reads Mol-Instructions, filters the description-guided molecule design task, decodes SELFIES targets to SMILES, canonicalizes with RDKit, deduplicates examples, and writes train/validation/test splits.

2. `scripts/train.py`

   Loads the split CSV/JSONL files, builds the selected SMILES representation, loads SmolLM, extends the tokenizer vocabulary if needed, fine-tunes with TRL `SFTTrainer`, saves the final model, and optionally runs validation generation.

3. `scripts/evaluate.py`

   Loads a trained run from `results/runs/<run-name>/`, generates SMILES for a split, detokenizes chemistry-specific representations back to raw SMILES, and computes RDKit-based metrics.

4. `scripts/run_hpo.py` and `scripts/summarize_hpo.py`

   Run matched hyperparameter sweeps per tokenizer strategy and summarize validation-selected best runs.

## Important Outputs

Each training run writes a directory under `results/runs/` containing:

```text
config.yaml
tokenizer_metrics.json
train_metrics.json
metrics.json
tokenizer/
final_model/
evaluation/
```

These outputs can be large and are ignored by Git. Keep them on local disk, Arrhenius project storage, or upload selected final artifacts to a model registry if needed.

## Local Smoke Test

After installing dependencies:

```bash
python scripts/prepare_data.py --config configs/default.yaml --set data.max_samples=128

python scripts/train.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/atomwise.yaml \
  --set dry_run=true \
  --set training.max_steps=2 \
  --set evaluation.max_validation_examples=4
```

## Legacy Folder

`legacy/mol_tokenizer_experiments/` is a self-contained earlier version of the same research idea. It uses its own package, requirement file, and Slurm scripts. Keep it if you want the old fixed-grid workflow, but use `src/`, `scripts/`, and `configs/` for the main project.
