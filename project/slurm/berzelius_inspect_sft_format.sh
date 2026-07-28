#!/usr/bin/env bash
set -euo pipefail

PROJECT_BASE="/proj/berzelius-2026-62/users/x_telcr"
REPO_DIR="$PROJECT_BASE/nlp-project"
PROJECT_DIR="$REPO_DIR/project"
IMAGE_PATH="$PROJECT_BASE/containers/pytorch-2.4.1-cuda12.4.sif"

cd "$PROJECT_DIR"

apptainer exec \
  --bind "$PROJECT_BASE:/work" \
  "$IMAGE_PATH" \
  bash -lc "export PYTHONPATH=/work/nlp-project/project:\${PYTHONPATH:-}; source /work/venvs/smollm/bin/activate && cd /work/nlp-project/project && python scripts/inspect_sft_format.py \
    --config configs/default.yaml \
    --tokenizer_config configs/tokenizers/atomwise.yaml \
    --examples 3 \
    --set model.model_id=HuggingFaceTB/SmolLM-135M-Instruct \
    --set data.sft_format=prompt_completion \
    --set data.use_chat_template=true \
    --set training.loss_type=nll \
    --set training.packing=false \
    --set training.padding_free=false \
    --set data.train_path=/work/results/data/mol_instructions_description_guided_short/train.csv"
