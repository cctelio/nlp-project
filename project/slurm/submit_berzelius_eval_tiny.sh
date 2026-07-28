#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

sbatch \
  --export=ALL,TOKENIZER_CONFIG=configs/tokenizers/atomwise.yaml,PER_DEVICE_TRAIN_BATCH_SIZE=32,GRADIENT_ACCUMULATION_STEPS=1,GRADIENT_CHECKPOINTING=false,PACKING=true,MAX_LENGTH=512,MAX_STEPS=10,CANONICALIZE_SMILES=true,EVALUATION_RUN_AFTER_TRAINING=true,EVALUATION_MAX_VALIDATION_EXAMPLES=8 \
  slurm/berzelius_short_train_pulled_image.sh
