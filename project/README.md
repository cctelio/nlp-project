# SmolLM Chemical Tokenization

Run all commands in this file from the `project/` directory.

If you have not tested the project yet, start with [docs/LOW_COMPUTE_RUNBOOK.md](docs/LOW_COMPUTE_RUNBOOK.md) before running the full experiments.

Cluster guides:

- [docs/BERZELIUS_CONTAINER.md](docs/BERZELIUS_CONTAINER.md)
- [docs/ARRHENIUS_CONTAINER.md](docs/ARRHENIUS_CONTAINER.md)

This repository compares tokenization strategies for small language models on chemical instruction data. The target task is natural-language-to-SMILES generation using the description-guided molecule design subset of Mol-Instructions.

The base model is `HuggingFaceTB/SmolLM-135M`. The main research question is whether chemical tokenization helps a small pretrained language model generate valid and target-similar SMILES without harming the natural-language side of the instruction-following task.

## Design Rationale

SmolLM already has a large pretrained vocabulary and useful natural-language tokenization. Because the input is a natural-language molecular description and the output is SMILES, replacing the whole tokenizer would confound the experiment: a new chemistry tokenizer may help SMILES but damage the model's pretrained language interface.

This codebase therefore preserves the base SmolLM tokenizer for all experiments. Chemical tokenizers add missing chemical tokens to the Hugging Face tokenizer vocabulary and resize the model embeddings before fine-tuning.

The specialized output representations are:

- `default`: raw SMILES, unmodified SmolLM tokenizer.
- `atomwise`: classical regex tokenized SMILES, trained as whitespace-separated atom/operator tokens and detokenized before RDKit evaluation.
- `smilespe`: SMILES Pair Encoding style tokens learned from training SMILES, also trained as whitespace-separated chemical tokens and detokenized before RDKit evaluation.

The atom-wise regex follows the common Schwaller/Molecular Transformer style pattern:

```python
r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
```

Because SmolLM's base vocabulary is already large, the actual number of new chemical tokens can be small. That does not make the comparison meaningless: the specialized runs also change the assistant-side representation, sequence lengths, and merge structure. The important question is whether those changes improve validation/test chemistry metrics enough to justify the extra representation complexity.

## Project Layout

```text
src/
  data/           # Mol-Instructions loading, filtering, deterministic splits
  tokenization/   # default, regex atom-wise, and SMILESPE representations
  training/       # TRL SFTTrainer workflow and HPO runner
  evaluation/     # generation, RDKit validity, exact match, Tanimoto metrics
  utils/          # config, logging, paths, reproducibility
configs/
  default.yaml
  tokenizers/
  sweeps/
scripts/
  prepare_data.py
  prepare_sft_data.py
  train.py
  evaluate.py
  run_hpo.py
  run_all_experiments.sh
results/
docs/
containers/
slurm/
legacy/
```

## Setup

Use a Python environment with CUDA-compatible PyTorch if training on GPU.

```bash
pip install -e ".[dev,tracking]"
```

RDKit is required for validity and Tanimoto metrics. If your package manager cannot resolve `rdkit` from PyPI, install it through conda/mamba and then install the rest of the project dependencies.

## Dataset Preparation

Prepare the Mol-Instructions description-guided molecule design subset:

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

By default the script downloads `data/Molecule-oriented_Instructions.zip` from the Hugging Face dataset `zjunlp/Mol-Instructions`. To use a local archive or extracted directory:

```bash
python scripts/prepare_data.py \
  --config configs/default.yaml \
  --set data.raw_path=/path/to/Molecule-oriented_Instructions.zip
```

The released `description_guided_molecule_design.json` file provides `metadata.split` values for `train` and `test`, but not validation. The preprocessing code preserves the official test split and deterministically carves validation examples from official train with seed `42`.

The released targets are SELFIES strings, not raw SMILES. Preprocessing decodes SELFIES to SMILES and canonicalizes the decoded SMILES with RDKit by default before writing `target_smiles`. It writes canonical question/answer splits as both CSV and JSONL, with CSV paths used by default:

```text
results/data/mol_instructions_description_guided/train.csv
results/data/mol_instructions_description_guided/validation.csv
results/data/mol_instructions_description_guided/test.csv
```

## Training One Run

Training reads the canonical CSV files, then builds tokenizer-specific SFT text inside the training process. This matches the phenoVLM pattern: preprocessing stores clean canonical examples, and the training script maps them into SFT records for the selected tokenizer strategy.

Default tokenizer:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/default.yaml
```

Atom-wise tokenizer:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/atomwise.yaml
```

SMILESPE tokenizer:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/smilespe.yaml
```

Each run writes to `results/runs/{tokenizer_strategy}-{run_id}/` and saves:

- `config.yaml`
- `tokenizer_metrics.json`
- `train_metrics.json`
- `metrics.json` when post-training validation generation is enabled
- `tokenizer/`
- `final_model/`
- `evaluation/validation_metrics.json`
- `evaluation/validation_generations.csv`

## Hyperparameter Optimization

For a fair tokenizer comparison, use the same HPO budget per tokenizer and select the best checkpoint by validation metrics before evaluating on the held-out test split.

Run one tokenizer sweep:

```bash
python scripts/run_hpo.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/default.yaml \
  --sweep_config configs/sweeps/default_hpo.yaml
```

Run all tokenizer sweeps:

```bash
bash scripts/run_all_experiments.sh
```

The default and atom-wise sweeps are 24-trial grids. The SMILESPE sweep samples 24 trials from a larger joint space so every tokenizer gets the same search budget.

The shared sweep dimensions are:

- learning rate
- number of epochs
- warmup ratio
- weight decay

The SMILESPE sweep additionally tunes:

- chemical vocabulary size
- minimum pair frequency

The primary selection metric is `validation/tanimoto_similarity_mean`. Tie-breakers should be validity, canonical exact-match accuracy, then validation loss.

Summarize validation-selected best runs:

```bash
python scripts/summarize_hpo.py
```

## Cluster Training From Local CSVs

You can do all SELFIES decoding and canonicalization locally, then copy only canonical question/answer CSVs to the cluster. The cluster training job applies the selected tokenizer strategy during SFT dataset construction.

Local preprocessing:

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

Copy these files/directories to the cluster:

```text
results/data/mol_instructions_description_guided/train.csv
results/data/mol_instructions_description_guided/validation.csv
results/data/mol_instructions_description_guided/test.csv
configs/
scripts/
src/
```

Cluster training only needs `torch`, `transformers`, `datasets`, `trl`, `pyyaml`, and optionally `wandb`. It does not need `selfies` for training from canonical CSVs. RDKit is only needed if `evaluation.run_after_training=true` or if you run `scripts/evaluate.py` on the cluster.

Example cluster training command:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/atomwise.yaml \
  --set data.train_path=/cluster/path/results/data/mol_instructions_description_guided/train.csv \
  --set data.validation_path=/cluster/path/results/data/mol_instructions_description_guided/validation.csv \
  --set data.test_path=/cluster/path/results/data/mol_instructions_description_guided/test.csv \
  --set logging.disable_wandb=false \
  --set training.output_root=/cluster/path/results/runs
```

For SMILESPE HPO, this same canonical CSV is reused for every trial. The SMILESPE merge rules are learned inside each training run from the canonical training SMILES, so you do not need trial-specific preprocessed CSVs.

After training, evaluate on a machine with RDKit, or directly on the cluster if RDKit is available:

```bash
python scripts/evaluate.py --run_dir results/runs/<run-name> --split test
```

## Evaluation

Evaluate a trained run on the test split:

```bash
python scripts/evaluate.py --run_dir results/runs/<run-name> --split test
```

After evaluating the selected runs on test, rerun:

```bash
python scripts/summarize_hpo.py
```

The summary CSV will include test metrics when `evaluation/test_metrics.json` exists in each best run directory.

Evaluation writes:

- `evaluation/test_metrics.json`
- `evaluation/test_generations.csv`

## Metrics

Tokenizer and vocabulary metrics:

- `base_vocab_size`
- `candidate_added_token_count`
- `added_token_count`
- `hf_add_tokens_return_count`
- `final_vocab_size`
- `vocab_expansion_fraction`
- train/validation/test average tokenized SMILES length
- train/validation/test median, q25, q75, and max tokenized SMILES length

Training metrics:

- training loss
- validation loss
- best/latest checkpoint
- configured learning rate, epochs, batch size, gradient accumulation, warmup, and weight decay through saved config

Generation and chemistry metrics:

- `exact_match_accuracy`
- `canonical_exact_match_accuracy`
- `validity`
- `invalid_rate`
- `empty_generation_rate`
- `tanimoto_similarity_mean`
- `tanimoto_similarity_q25`
- `tanimoto_similarity_median`
- `tanimoto_similarity_q75`
- target tokenized SMILES length summary
- generated tokenized SMILES length summary
- qualitative generation examples in CSV form

Tanimoto similarity uses RDKit Morgan fingerprints with radius `2` and `2048` bits. Invalid generations receive similarity `0.0`.

## Reproducibility

The project sets Python, NumPy, PyTorch, CUDA, and Transformers seeds where available. Data splits, SMILESPE learning, and generation seeds are deterministic. Every run saves the merged config used for that experiment.

## Dry Run

To smoke-test the pipeline on tiny splits:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/atomwise.yaml \
  --set dry_run=true \
  --set training.max_steps=2 \
  --set evaluation.max_validation_examples=4
```

## Notes

- The default tokenizer preserves SmolLM exactly.
- Atom-wise and SMILESPE runs build represented SMILES responses inside the training script, add chemical tokens with Hugging Face tokenizer APIs, and resize model embeddings before training.
- Specialized tokenizer generations are detokenized before RDKit evaluation, so all strategies are evaluated as raw SMILES.
