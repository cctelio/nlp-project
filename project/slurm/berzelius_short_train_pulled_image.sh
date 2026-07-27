#!/usr/bin/env bash
#SBATCH -A berzelius-2026-62
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J smollm-pull-short
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

PROJECT_BASE="/proj/berzelius-2026-62/users/x_telcr"
REPO_DIR="$PROJECT_BASE/nlp-project"
PROJECT_DIR="$REPO_DIR/project"
IMAGE_PATH="$PROJECT_BASE/containers/pytorch-2.4.1-cuda12.4.sif"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-configs/tokenizers/atomwise.yaml}"

mkdir -p \
  "$PROJECT_BASE/hf_cache" \
  "$PROJECT_BASE/hf_datasets_cache" \
  "$PROJECT_BASE/results/short_runs" \
  "$PROJECT_BASE/results/data" \
  "$PROJECT_BASE/triton_cache" \
  "$PROJECT_BASE/wandb" \
  "$PROJECT_BASE/wandb_cache"

cd "$PROJECT_DIR"

COMMON_ENV="export HF_HOME=/work/hf_cache; export HF_DATASETS_CACHE=/work/hf_datasets_cache; export WANDB_DIR=/work/wandb; export WANDB_CACHE_DIR=/work/wandb_cache; export TRITON_CACHE_DIR=/work/triton_cache; export TOKENIZERS_PARALLELISM=false"

apptainer exec --nv \
  --bind "$PROJECT_BASE:/work" \
  "$IMAGE_PATH" \
  bash -lc "$COMMON_ENV; source /work/venvs/smollm/bin/activate && cd /work/nlp-project/project && python scripts/prepare_data.py \
    --config configs/default.yaml \
    --set data.max_samples=1024 \
    --set data.cache_dir=/work/hf_cache/mol_instructions \
    --set data.processed_dir=/work/results/data/mol_instructions_description_guided_short \
    --set data.train_path=/work/results/data/mol_instructions_description_guided_short/train.csv \
    --set data.validation_path=/work/results/data/mol_instructions_description_guided_short/validation.csv \
    --set data.test_path=/work/results/data/mol_instructions_description_guided_short/test.csv"

apptainer exec --nv \
  --bind "$PROJECT_BASE:/work" \
  "$IMAGE_PATH" \
  bash -lc "$COMMON_ENV; source /work/venvs/smollm/bin/activate && cd /work/nlp-project/project && python scripts/train.py \
    --config configs/default.yaml \
    --tokenizer_config '$TOKENIZER_CONFIG' \
    --set training.max_steps=50 \
    --set training.output_root=/work/results/short_runs \
    --set training.save_total_limit=1 \
    --set evaluation.run_after_training=false \
    --set generation.batch_size=4 \
    --set data.cache_dir=/work/hf_cache/mol_instructions \
    --set data.processed_dir=/work/results/data/mol_instructions_description_guided_short \
    --set data.train_path=/work/results/data/mol_instructions_description_guided_short/train.csv \
    --set data.validation_path=/work/results/data/mol_instructions_description_guided_short/validation.csv \
    --set data.test_path=/work/results/data/mol_instructions_description_guided_short/test.csv \
    --set logging.disable_wandb=true"
