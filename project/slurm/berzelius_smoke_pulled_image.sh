#!/usr/bin/env bash
#SBATCH -A berzelius-2026-62
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -J smollm-pull-smoke
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

PROJECT_BASE="/proj/berzelius-2026-62/users/x_telcr"
REPO_DIR="$PROJECT_BASE/nlp-project"
PROJECT_DIR="$REPO_DIR/project"
IMAGE_PATH="$PROJECT_BASE/containers/pytorch-2.4.1-cuda12.4.sif"
VENV_PATH="$PROJECT_BASE/venvs/smollm"

mkdir -p \
  "$PROJECT_BASE/hf_cache" \
  "$PROJECT_BASE/hf_datasets_cache" \
  "$PROJECT_BASE/results/smoke_runs" \
  "$PROJECT_BASE/wandb" \
  "$PROJECT_BASE/wandb_cache"

cd "$PROJECT_DIR"

apptainer exec --nv \
  --bind "$PROJECT_BASE:/work" \
  "$IMAGE_PATH" \
  bash -lc "source /work/venvs/smollm/bin/activate && cd /work/nlp-project/project && python scripts/make_tiny_smoke_data.py \
    --output_dir /work/results/data/tiny_smoke"

apptainer exec --nv \
  --bind "$PROJECT_BASE:/work" \
  "$IMAGE_PATH" \
  bash -lc "source /work/venvs/smollm/bin/activate && cd /work/nlp-project/project && python scripts/train.py \
    --config configs/default.yaml \
    --tokenizer_config configs/tokenizers/atomwise.yaml \
    --set dry_run=true \
    --set training.max_steps=2 \
    --set training.output_root=/work/results/smoke_runs \
    --set training.save_total_limit=1 \
    --set evaluation.run_after_training=false \
    --set data.train_path=/work/results/data/tiny_smoke/train.csv \
    --set data.validation_path=/work/results/data/tiny_smoke/validation.csv \
    --set data.test_path=/work/results/data/tiny_smoke/test.csv \
    --set logging.disable_wandb=true"
