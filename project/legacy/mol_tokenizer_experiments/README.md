# Mol-Instructions Tokenizer Experiments

Self-contained fixed-grid experiments for evaluating SMILES tokenization strategies on the Mol-Instructions **description-guided molecule design** task.

The core comparison uses `HuggingFaceTB/SmolLM-135M-Instruct` with identical training hyperparameters across tokenizer strategies:

- `default_bpe`: unchanged base tokenizer, raw SMILES targets.
- `atomwise`: regex SMILES atom/token splitting, space-delimited targets, vocabulary expansion.
- `smilespe_1000`: SMILES pair encoding with 1000 merge operations, trained on train SMILES only.
- `smilespe_2000`: SMILES pair encoding with 2000 merge operations, trained on train SMILES only.

Evaluation reports raw exact match, canonical SMILES exact match, RDKit validity, Morgan fingerprint Tanimoto similarity, average target token length, and added vocabulary size.

## Mac Smoke Test

Use this before submitting the Slurm grid. It runs a tiny subset through the full path:
dataset loading, tokenizer strategy fitting, vocabulary expansion, SFT, generation, and RDKit metrics.

```bash
cd /Users/telio/nlp-project-VLM
conda activate phenoVLM-env

python -c "import torch; print(torch.__version__); print('mps:', torch.backends.mps.is_available())"

mol_tokenizer_experiments/scripts/run_local_mps_smoke.sh
```

The training script prints `Compute backend: mps` when PyTorch can use Apple Silicon. If it prints `cpu`, the same smoke test still works but will be slower.

Useful overrides:

```bash
TOKENIZER_STRATEGY=default_bpe MAX_TRAIN_SAMPLES=16 MAX_EVAL_SAMPLES=8 \
  mol_tokenizer_experiments/scripts/run_local_mps_smoke.sh
```

## Cluster Fixed Grid

Edit paths in `mol_tokenizer_experiments/slurm/submit_tokenizer_grid.sh` if needed, then run:

```bash
cd /cephyr/users/telio/nlp-project-VLM
bash mol_tokenizer_experiments/slurm/submit_tokenizer_grid.sh
```

The Slurm array runs one tokenizer strategy per task.

## Dataset

The loader uses:

- Dataset id: `zjunlp/Mol-Instructions`
- File: `data/Molecule-oriented_Instructions.zip`
- Task JSON: `Molecule-oriented_Instructions/description_guided_molecule_design.json`

The code downloads the zip directly from the Hugging Face dataset repository instead of using the legacy dataset loading script.
