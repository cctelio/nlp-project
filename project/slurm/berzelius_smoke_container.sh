#!/usr/bin/env bash
#SBATCH -A <berzelius_project_account>
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -J smollm-smoke
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

# Edit these two values on Berzelius.
PROJECT_BASE="/proj/<berzelius_project>/users/telio"
REPO_DIR="$PROJECT_BASE/nlp-project"

PROJECT_DIR="$REPO_DIR/project"
CONTAINER_PATH="$PROJECT_BASE/containers/smollm-chemical-tokenization.sif"

mkdir -p \
  "$PROJECT_BASE/hf_cache" \
  "$PROJECT_BASE/hf_datasets_cache" \
  "$PROJECT_BASE/wandb" \
  "$PROJECT_BASE/wandb_cache" \
  "$PROJECT_BASE/results/smoke_runs"

cd "$PROJECT_DIR"

apptainer exec --nv \
  --bind "$PROJECT_BASE:/work" \
  "$CONTAINER_PATH" \
  bash -lc "cd /work/nlp-project/project && python scripts/prepare_data.py \
    --config configs/default.yaml \
    --set data.max_samples=128 \
    --set data.cache_dir=/work/hf_cache/mol_instructions \
    --set data.processed_dir=/work/results/data/mol_instructions_description_guided_smoke \
    --set data.train_path=/work/results/data/mol_instructions_description_guided_smoke/train.csv \
    --set data.validation_path=/work/results/data/mol_instructions_description_guided_smoke/validation.csv \
    --set data.test_path=/work/results/data/mol_instructions_description_guided_smoke/test.csv"

apptainer exec --nv \
  --bind "$PROJECT_BASE:/work" \
  "$CONTAINER_PATH" \
  bash -lc "cd /work/nlp-project/project && python scripts/train.py \
    --config configs/default.yaml \
    --tokenizer_config configs/tokenizers/atomwise.yaml \
    --set dry_run=true \
    --set training.max_steps=2 \
    --set training.output_root=/work/results/smoke_runs \
    --set training.save_total_limit=1 \
    --set evaluation.max_validation_examples=4 \
    --set generation.batch_size=4 \
    --set data.cache_dir=/work/hf_cache/mol_instructions \
    --set data.processed_dir=/work/results/data/mol_instructions_description_guided_smoke \
    --set data.train_path=/work/results/data/mol_instructions_description_guided_smoke/train.csv \
    --set data.validation_path=/work/results/data/mol_instructions_description_guided_smoke/validation.csv \
    --set data.test_path=/work/results/data/mol_instructions_description_guided_smoke/test.csv \
    --set logging.disable_wandb=true"
