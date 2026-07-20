#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py --config configs/default.yaml

python scripts/run_hpo.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/default.yaml \
  --sweep_config configs/sweeps/default_hpo.yaml

python scripts/run_hpo.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/atomwise.yaml \
  --sweep_config configs/sweeps/atomwise_hpo.yaml

python scripts/run_hpo.py \
  --config configs/default.yaml \
  --tokenizer_config configs/tokenizers/smilespe.yaml \
  --sweep_config configs/sweeps/smilespe_hpo.yaml

python scripts/summarize_hpo.py
