#!/usr/bin/env bash
set -euo pipefail

WRAPPER="slurm/berzelius_short_train_pulled_image.sh"

COMMON_EXPORTS="MODEL_ID=HuggingFaceTB/SmolLM-135M-Instruct"
COMMON_EXPORTS+=",SFT_FORMAT=prompt_completion"
COMMON_EXPORTS+=",USE_CHAT_TEMPLATE=true"
COMMON_EXPORTS+=",DATA_MAX_SAMPLES=8192"
COMMON_EXPORTS+=",LEARNING_RATE=0.00005"
COMMON_EXPORTS+=",GRADIENT_CHECKPOINTING=false"
COMMON_EXPORTS+=",PACKING=false"
COMMON_EXPORTS+=",LOSS_TYPE=nll"
COMMON_EXPORTS+=",LOAD_BEST_MODEL_AT_END=true"
COMMON_EXPORTS+=",CANONICALIZE_SMILES=true"
COMMON_EXPORTS+=",EVALUATION_RUN_AFTER_TRAINING=true"
COMMON_EXPORTS+=",EVALUATION_SPLIT=validation"
COMMON_EXPORTS+=",EVALUATION_MAX_VALIDATION_EXAMPLES=128"

submit_run() {
  local job_name="$1"
  local tokenizer_config="$2"
  local max_length="$3"
  local batch_size="$4"
  local grad_accum="$5"
  local max_steps="$6"
  local gen_tokens="$7"

  local exports="ALL,${COMMON_EXPORTS}"
  exports+=",TOKENIZER_CONFIG=${tokenizer_config}"
  exports+=",MAX_LENGTH=${max_length}"
  exports+=",PER_DEVICE_TRAIN_BATCH_SIZE=${batch_size}"
  exports+=",GRADIENT_ACCUMULATION_STEPS=${grad_accum}"
  exports+=",MAX_STEPS=${max_steps}"
  exports+=",GENERATION_MAX_NEW_TOKENS=${gen_tokens}"

  echo "Submitting ${job_name}"
  sbatch --job-name="$job_name" --export="$exports" "$WRAPPER"
}

# Retry SMILESPE first. The earlier JSON error was likely from concurrent jobs
# sharing the same extracted Mol-Instructions cache.
submit_run smollm-smilespe-l1024 configs/tokenizers/smilespe.yaml 1024 8 2 4000 128

# Same clean setup, but with longer context. Batch is reduced to fit memory.
submit_run smollm-default-l2048 configs/tokenizers/default.yaml 2048 4 4 4000 128
submit_run smollm-atomwise-l2048 configs/tokenizers/atomwise.yaml 2048 4 4 4000 128
