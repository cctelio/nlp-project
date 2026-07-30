#!/usr/bin/env bash
set -euo pipefail

WRAPPER="slurm/berzelius_short_train_pulled_image.sh"

COMMON_EXPORTS="MODEL_ID=HuggingFaceTB/SmolLM-135M-Instruct"
COMMON_EXPORTS+=",TOKENIZER_CONFIG=configs/tokenizers/default.yaml"
COMMON_EXPORTS+=",SFT_FORMAT=prompt_completion"
COMMON_EXPORTS+=",USE_CHAT_TEMPLATE=true"
COMMON_EXPORTS+=",DATA_MAX_SAMPLES=0"
COMMON_EXPORTS+=",MAX_LENGTH=1024"
COMMON_EXPORTS+=",PER_DEVICE_TRAIN_BATCH_SIZE=8"
COMMON_EXPORTS+=",GRADIENT_ACCUMULATION_STEPS=2"
COMMON_EXPORTS+=",GRADIENT_CHECKPOINTING=false"
COMMON_EXPORTS+=",PACKING=false"
COMMON_EXPORTS+=",LOSS_TYPE=nll"
COMMON_EXPORTS+=",LOAD_BEST_MODEL_AT_END=false"
COMMON_EXPORTS+=",SAVE_TOTAL_LIMIT=2"
COMMON_EXPORTS+=",CANONICALIZE_SMILES=true"
COMMON_EXPORTS+=",EVALUATION_RUN_AFTER_TRAINING=true"
COMMON_EXPORTS+=",EVALUATION_SPLIT=validation"
COMMON_EXPORTS+=",EVALUATION_MAX_VALIDATION_EXAMPLES=128"
COMMON_EXPORTS+=",GENERATION_MAX_NEW_TOKENS=128"

submit_lr() {
  local lr="$1"
  local job_name="$2"
  local exports="ALL,${COMMON_EXPORTS},LEARNING_RATE=${lr},MAX_STEPS=4000"

  echo "Submitting ${job_name} with lr=${lr}"
  sbatch --job-name="$job_name" --time=08:00:00 --export="$exports" "$WRAPPER"
}

submit_lr 0.00001 smollm-default-full-lr1e-5
submit_lr 0.00002 smollm-default-full-lr2e-5
submit_lr 0.00005 smollm-default-full-lr5e-5
