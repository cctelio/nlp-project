#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-462
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -N 1
#SBATCH -t 0-00:30:00
#SBATCH -J "mol-token-smoke"

set -euo pipefail

PERSISTENT_DIR="${PERSISTENT_DIR:-/mimer/NOBACKUP/groups/naiss2023-6-290/telio}"
CONTAINER_PATH="${CONTAINER_PATH:-/cephyr/users/telio/nanoVLM/my_container.sif}"
PROJECT_DIR="${PROJECT_DIR:-/cephyr/users/telio/nlp-project-VLM}"
OUTPUT_DIR="${OUTPUT_DIR:-$PERSISTENT_DIR/mol_tokenizer_experiments_smoke}"

export HF_HOME="$PERSISTENT_DIR/hf_cache"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PERSISTENT_DIR/hf_datasets_cache}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

srun apptainer exec --nv "$CONTAINER_PATH" bash -lc "
  cd '$PROJECT_DIR' &&
  python mol_tokenizer_experiments/scripts/train_tokenizer_experiment.py \
    --tokenizer_strategy atomwise \
    --dry_run \
    --disable_wandb \
    --output_dir '$OUTPUT_DIR' \
    --processed_dir '$OUTPUT_DIR/data/processed/molinstructions'
"
