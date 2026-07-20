#!/usr/bin/env bash
#SBATCH -A <project>
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -J smollm-chem
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

# Edit these paths after cloning the repository on Arrhenius.
PROJECT_BASE="${PROJECT_BASE:-/nobackup/proj/disk/<project>/telio}"
REPO_DIR="${REPO_DIR:-$PROJECT_BASE/nlp-project}"
PROJECT_DIR="${PROJECT_DIR:-$REPO_DIR/project}"
CONTAINER_PATH="${CONTAINER_PATH:-$PROJECT_BASE/containers/smollm-chemical-tokenization.sif}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_BASE/results}"

TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-configs/tokenizers/atomwise.yaml}"

export HF_HOME="${HF_HOME:-$PROJECT_BASE/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_BASE/hf_datasets_cache}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_BASE/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PROJECT_BASE/wandb_cache}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$RESULTS_DIR"

cd "$PROJECT_DIR"

srun apptainer exec --nv \
  --bind "$PROJECT_BASE:/work" \
  "$CONTAINER_PATH" \
  bash -lc "cd /work/nlp-project/project && python scripts/train.py \
    --config configs/default.yaml \
    --tokenizer_config '$TOKENIZER_CONFIG' \
    --set data.cache_dir=/work/hf_cache/mol_instructions \
    --set data.processed_dir=/work/results/data/mol_instructions_description_guided \
    --set data.train_path=/work/results/data/mol_instructions_description_guided/train.csv \
    --set data.validation_path=/work/results/data/mol_instructions_description_guided/validation.csv \
    --set data.test_path=/work/results/data/mol_instructions_description_guided/test.csv \
    --set training.output_root=/work/results/runs \
    --set logging.disable_wandb=true"
