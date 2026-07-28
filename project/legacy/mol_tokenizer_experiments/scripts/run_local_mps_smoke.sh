#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

python mol_tokenizer_experiments/scripts/train_tokenizer_experiment.py \
  --tokenizer_strategy "${TOKENIZER_STRATEGY:-atomwise}" \
  --dry_run \
  --disable_wandb \
  --max_train_samples "${MAX_TRAIN_SAMPLES:-8}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-4}" \
  --num_eval_generations "${NUM_EVAL_GENERATIONS:-4}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --eval_generation_batch_size "${GEN_BATCH_SIZE:-1}" \
  --max_length "${MAX_LENGTH:-256}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-96}" \
  --output_dir "${OUTPUT_DIR:-mol_tokenizer_experiments/outputs/local_mps_smoke}" \
  "$@"
