# NLP Course Repository

This repository is intentionally split into two top-level areas.

```text
assignments/   # Four course assignment notebooks and assignment notes
project/       # Main NLP project: SmolLM chemical tokenization
```

## Assignments

Open [assignments/README.md](assignments/README.md) for the assignment notebook notes.

The four notebooks are:

```text
assignments/assignment_1_language_modeling.ipynb
assignments/assignment_2_transformer_language_models.ipynb
assignments/WASP_NLP_A3_skeleton.ipynb
assignments/assignment_4_retrieval_augmented_generation.ipynb
```

Generated assignment data and checkpoints are ignored by Git.

## Main NLP Project

Open [project/README.md](project/README.md) for the project instructions.

Useful project docs:

```text
project/docs/PROJECT_ARCHITECTURE.md
project/docs/ARRHENIUS_CONTAINER.md
project/docs/LOW_COMPUTE_RUNBOOK.md
```

Run project commands from the `project/` directory:

```bash
cd project
python scripts/prepare_data.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --tokenizer_config configs/tokenizers/atomwise.yaml
```

## GitHub Upload Rule

Commit source, configs, docs, tests, and notebooks. Do not commit generated data, caches, model checkpoints, or experiment outputs.
