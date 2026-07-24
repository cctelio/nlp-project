# Low-Compute Runbook

Use this when you want to understand and test the project before spending meaningful cluster time or storage.

Run every command from the `project/` directory.

## How the Project Gets Data

The project uses the Hugging Face dataset:

```text
zjunlp/Mol-Instructions
```

Specifically, it downloads:

```text
data/Molecule-oriented_Instructions.zip
```

from that dataset repository. This is configured in:

```text
configs/default.yaml
```

The preprocessing script is:

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

What preprocessing does:

1. Downloads or reads the Mol-Instructions archive.
2. Extracts molecule-oriented instruction records.
3. Filters the `description-guided molecule design` task.
4. Reads each instruction and molecular target.
5. Detects whether the target is SELFIES or SMILES.
6. Decodes SELFIES to SMILES when needed.
7. Canonicalizes SMILES with RDKit.
8. Deduplicates examples.
9. Preserves the official test split.
10. Creates a deterministic validation split from official train when no official validation split exists.
11. Writes canonical train/validation/test files.

Default output:

```text
results/data/mol_instructions_description_guided/train.csv
results/data/mol_instructions_description_guided/validation.csv
results/data/mol_instructions_description_guided/test.csv
```

These files are generated artifacts and should not be committed.

## Expected Pipeline

The intended full pipeline is:

```text
prepare_data.py
  -> train.py with default tokenizer
  -> train.py with atomwise tokenizer
  -> train.py with smilespe tokenizer
  -> evaluate.py on selected trained runs
  -> summarize_hpo.py if using HPO sweeps
```

For one single run:

```bash
python scripts/prepare_data.py --config configs/default.yaml

python scripts/train.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/atomwise.yaml

python scripts/evaluate.py \
  --run_dir results/runs/<run-name> \
  --split test
```

For the final comparison, run the same budget for:

```text
configs/tokenizers/default.yaml
configs/tokenizers/atomwise.yaml
configs/tokenizers/smilespe.yaml
```

## Safe First Test

Do this locally or in a short interactive cluster job. It prepares only a tiny sample and trains for two steps.

```bash
python scripts/prepare_data.py \
  --config configs/default.yaml \
  --set data.max_samples=128

python scripts/train.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/atomwise.yaml \
  --set dry_run=true \
  --set training.max_steps=2 \
  --set training.output_root=results/smoke_runs \
  --set evaluation.max_validation_examples=4 \
  --set logging.disable_wandb=true
```

Expected output:

```text
results/smoke_runs/<run-name>/
```

This proves that data loading, tokenizer construction, model loading, SFT setup, generation, and RDKit metrics work.

## Storage-Safe Settings

For first runs, keep:

```yaml
training.save_total_limit: 1
evaluation.max_validation_examples: 16
logging.disable_wandb: true
```

Useful command-line overrides:

```bash
--set training.save_total_limit=1
--set evaluation.max_validation_examples=16
--set generation.batch_size=4
--set training.output_root=/nobackup/proj/disk/<project>/<user>/results/smoke_runs
```

Avoid running HPO until one dry run and one short real run have completed.

## Suggested Progression

1. Tiny dry run:

   ```bash
   --set dry_run=true --set training.max_steps=2
   ```

2. Short single-tokenizer run:

   ```bash
   --set data.max_samples=1024
   --set training.max_steps=50
   --set evaluation.max_validation_examples=16
   ```

3. One realistic run for `atomwise`.

4. Repeat the same realistic settings for `default` and `smilespe`.

5. Only then run HPO.

## Container Build Note for Arrhenius

Arrhenius GPU nodes use NVIDIA Grace Hopper, which means the host CPU architecture is ARM64. If Apptainer reports:

```text
FATAL: image targets 'amd64', cannot run on 'arm64'
```

the container recipe pulled an x86/AMD64 image. Build from `containers/smollm-chemical-tokenization.def`, which uses an NVIDIA NGC PyTorch base image intended for NVIDIA GPU systems, and build it on an Arrhenius GPU node.

## Berzelius First-Run Path

If you have access to Berzelius, it is a good first target for this project because it is AI/ML-focused and supports Apptainer directly. Use:

```text
docs/BERZELIUS_CONTAINER.md
containers/smollm-chemical-tokenization-berzelius.def
slurm/berzelius_smoke_container.sh
slurm/berzelius_short_train_container.sh
```

Run the smoke script before any HPO or full training.

## W&B Setup

The code disables W&B by default:

```yaml
logging:
  disable_wandb: true
```

Enable it with:

```bash
--set logging.disable_wandb=false
--set logging.wandb_project=SmolLM-chemical-tokenization
```

On a cluster, avoid putting secrets in files committed to Git. Use environment variables in your shell or job environment:

```bash
export WANDB_API_KEY=<your_api_key>
export WANDB_PROJECT=SmolLM-chemical-tokenization
export WANDB_DIR=/nobackup/proj/disk/<project>/<user>/wandb
export WANDB_CACHE_DIR=/nobackup/proj/disk/<project>/<user>/wandb_cache
```

For a first cluster test where you want W&B logs but no network sync:

```bash
export WANDB_MODE=offline
```

Later, sync from the cluster login node or another machine with network access:

```bash
wandb sync --sync-all /nobackup/proj/disk/<project>/<user>/wandb
```

## What Not to Do First

Do not start with:

```bash
bash scripts/run_all_experiments.sh
```

That runs all sweeps and can spend much more compute and storage than needed.

Do not commit:

```text
results/
data/
*.safetensors
*.pt
wandb/
```
