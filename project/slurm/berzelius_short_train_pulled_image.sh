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
MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolLM-135M}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-configs/tokenizers/atomwise.yaml}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
MAX_STEPS="${MAX_STEPS:-50}"
LEARNING_RATE="${LEARNING_RATE:-0.00002}"
PACKING="${PACKING:-false}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CANONICALIZE_SMILES="${CANONICALIZE_SMILES:-false}"
EVALUATION_RUN_AFTER_TRAINING="${EVALUATION_RUN_AFTER_TRAINING:-false}"
EVALUATION_MAX_VALIDATION_EXAMPLES="${EVALUATION_MAX_VALIDATION_EXAMPLES:-16}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-160}"
SFT_FORMAT="${SFT_FORMAT:-text}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-false}"

mkdir -p \
  "$PROJECT_BASE/hf_cache" \
  "$PROJECT_BASE/hf_datasets_cache" \
  "$PROJECT_BASE/results/gpu_logs" \
  "$PROJECT_BASE/results/short_runs" \
  "$PROJECT_BASE/results/data" \
  "$PROJECT_BASE/triton_cache" \
  "$PROJECT_BASE/wandb" \
  "$PROJECT_BASE/wandb_cache"

cd "$PROJECT_DIR"

COMMON_ENV="export HF_HOME=/work/hf_cache; export HF_DATASETS_CACHE=/work/hf_datasets_cache; export WANDB_DIR=/work/wandb; export WANDB_CACHE_DIR=/work/wandb_cache; export TRITON_CACHE_DIR=/work/triton_cache; export TOKENIZERS_PARALLELISM=false"

GPU_LOG="$PROJECT_BASE/results/gpu_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.csv"
(
  echo "timestamp,index,name,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,power_draw_w"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits
    sleep 10
  done
) > "$GPU_LOG" 2>/dev/null &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

apptainer exec --nv \
  --bind "$PROJECT_BASE:/work" \
  "$IMAGE_PATH" \
  bash -lc "$COMMON_ENV; source /work/venvs/smollm/bin/activate && cd /work/nlp-project/project && python scripts/prepare_data.py \
    --config configs/default.yaml \
    --set data.max_samples=1024 \
    --set data.canonicalize_smiles='$CANONICALIZE_SMILES' \
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
    --set model.model_id='$MODEL_ID' \
    --set data.sft_format='$SFT_FORMAT' \
    --set data.use_chat_template='$USE_CHAT_TEMPLATE' \
    --set training.max_steps='$MAX_STEPS' \
    --set training.learning_rate='$LEARNING_RATE' \
    --set training.per_device_train_batch_size='$PER_DEVICE_TRAIN_BATCH_SIZE' \
    --set training.gradient_accumulation_steps='$GRADIENT_ACCUMULATION_STEPS' \
    --set training.gradient_checkpointing='$GRADIENT_CHECKPOINTING' \
    --set training.packing='$PACKING' \
    --set training.max_length='$MAX_LENGTH' \
    --set training.output_root=/work/results/short_runs \
    --set training.save_total_limit=1 \
    --set evaluation.run_after_training='$EVALUATION_RUN_AFTER_TRAINING' \
    --set evaluation.max_validation_examples='$EVALUATION_MAX_VALIDATION_EXAMPLES' \
    --set generation.max_new_tokens='$GENERATION_MAX_NEW_TOKENS' \
    --set generation.batch_size=4 \
    --set data.cache_dir=/work/hf_cache/mol_instructions \
    --set data.processed_dir=/work/results/data/mol_instructions_description_guided_short \
    --set data.train_path=/work/results/data/mol_instructions_description_guided_short/train.csv \
    --set data.validation_path=/work/results/data/mol_instructions_description_guided_short/validation.csv \
    --set data.test_path=/work/results/data/mol_instructions_description_guided_short/test.csv \
    --set logging.disable_wandb=true"

echo "GPU usage log: $GPU_LOG"
