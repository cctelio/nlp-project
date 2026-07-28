#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-462
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:4
#SBATCH -N 1
#SBATCH -t 0-06:00:00
#SBATCH -J "mol-token-grid"
#SBATCH --array=0-3

set -euo pipefail

STRATEGIES=(default_bpe atomwise smilespe_1000 smilespe_2000)
TOKENIZER_STRATEGY="${TOKENIZER_STRATEGY:-${STRATEGIES[$SLURM_ARRAY_TASK_ID]}}"

PERSISTENT_DIR="${PERSISTENT_DIR:-/mimer/NOBACKUP/groups/naiss2023-6-290/telio}"
CONTAINER_PATH="${CONTAINER_PATH:-/cephyr/users/telio/nanoVLM/my_container.sif}"
PROJECT_DIR="${PROJECT_DIR:-/cephyr/users/telio/nlp-project-VLM}"
OUTPUT_DIR="${OUTPUT_DIR:-$PERSISTENT_DIR/mol_tokenizer_experiments}"

export HF_HOME="$PERSISTENT_DIR/hf_cache"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PERSISTENT_DIR/hf_datasets_cache}"
export WANDB_DIR="$PERSISTENT_DIR/wandb_logs"
export WANDB_CACHE_DIR="$PERSISTENT_DIR/wandb_cache"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$OUTPUT_DIR"

cd "$PROJECT_DIR"

srun apptainer exec --nv "$CONTAINER_PATH" bash -lc "
  cd '$PROJECT_DIR' &&
  torchrun \
    --node_rank=\"\$SLURM_NODEID\" \
    --nnodes=\"\$SLURM_JOB_NUM_NODES\" \
    --nproc_per_node=\"\$SLURM_GPUS_ON_NODE\" \
    --rdzv_id=\"\$SLURM_JOB_ID\" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\"\$PROEPI_HEAD_NODE:\$PROEPI_FREE_PORT\" \
    mol_tokenizer_experiments/scripts/train_tokenizer_experiment.py \
    --tokenizer_strategy '$TOKENIZER_STRATEGY' \
    --output_dir '$OUTPUT_DIR' \
    --processed_dir '$OUTPUT_DIR/data/processed/molinstructions' \
    ${EXTRA_ARGS:-}
"
