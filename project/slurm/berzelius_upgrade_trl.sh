#!/usr/bin/env bash
set -euo pipefail

PROJECT_BASE="/proj/berzelius-2026-62/users/x_telcr"
IMAGE_PATH="$PROJECT_BASE/containers/pytorch-2.4.1-cuda12.4.sif"

apptainer exec \
  --bind "$PROJECT_BASE:/work" \
  "$IMAGE_PATH" \
  bash -lc "source /work/venvs/smollm/bin/activate && cd /work/nlp-project/project && python -m pip install --upgrade '.[tracking]' && python -c 'import trl, transformers; print(\"trl\", trl.__version__); print(\"transformers\", transformers.__version__)'"
