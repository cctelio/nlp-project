#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/cephyr/users/telio/nlp-project-VLM}"
SBATCH_SCRIPT="$PROJECT_DIR/mol_tokenizer_experiments/slurm/run_tokenizer_grid.sh"
CONTAINER_PATH="${CONTAINER_PATH:-/cephyr/users/telio/nanoVLM/my_container.sif}"
OUTPUT_DIR="${OUTPUT_DIR:-/mimer/NOBACKUP/groups/naiss2023-6-290/telio/mol_tokenizer_experiments}"

sbatch \
  --array=0-3 \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONTAINER_PATH="$CONTAINER_PATH",OUTPUT_DIR="$OUTPUT_DIR" \
  "$SBATCH_SCRIPT"
