# Course Assignment Notebooks

This folder is separate from the main NLP project in the repository root.

The four assignment notebooks are:

```text
assignment_1_language_modeling.ipynb
assignment_2_transformer_language_models.ipynb
WASP_NLP_A3_skeleton.ipynb
assignment_4_retrieval_augmented_generation.ipynb
```

Generated assignment data and trained assignment models should stay out of Git:

```text
assignments/data/
assignments/results/
```

These notebooks are meant to be run from the repository root.

## Environment

Use the same environment as the main project if convenient:

```bash
conda activate /Users/telio/miniconda3/envs/phenoVLM-env
python -m pip install nltk
python -m nltk.downloader punkt punkt_tab
python -m ipykernel install --user --name nlp-project --display-name "Python (nlp-project)"
```

Minimum useful packages are `torch`, `transformers`, `datasets`, `notebook` or `jupyterlab`, and `nltk`.

## Which Dataset?

There are two independent datasets in this repository:

- Main chemistry project: use Mol-Instructions, only the description-guided molecule design subset. This is for SmolLM natural-language-to-SMILES fine-tuning.
- Assignment notebooks: use the official Assignment 1 Wikipedia paragraph archive from the course page. Assignment 2 reuses this same text dataset and changes the model architecture from RNN to Transformer.

## Assignment Data

`assignment_1_language_modeling.ipynb` downloads and extracts the official Assignment 1 data automatically if needed. After the first setup cell runs, these files should exist:

```text
data/a1_1/train.txt
data/a1_1/val.txt
```

There is no fallback corpus. If the download or extraction fails, the notebook stops with an error instead of training on sample text.

For development, the notebook keeps `USE_SMALL_DATASET = True` in the HuggingFace dataset-loading cell, following the assignment suggestion to work with a subset while checking that the code runs. Set it to `False` for the full run.

Training length and batch sizes are controlled in the Task 4.1 cell:

```python
TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 3e-3
```

Change these values before running the trainer. If you want a fresh model for a new setting, rerun the Task 3.1 model setup cell first.

`assignment_2_transformer_language_models.ipynb` reuses the Assignment 1 text files from `data/a1_1`, as requested by the assignment page. It uses the same automatic data setup and has no fallback corpus. It also reuses the Assignment 1-style tokenizer settings, including `MAX_VOC_SIZE`, NLTK word splitting, padded paragraph batches, and an Assignment-1-style trainer with editable epoch and batch-size settings.

Assignment 2 defaults:

```python
TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 48
LEARNING_RATE = 2e-3
```

## Assignment 3

`WASP_NLP_A3_skeleton.ipynb` is the fine-tuning assignment. It loads SmolTalk from Hugging Face, formats prompt/response pairs, masks prompt tokens with `-100`, evaluates the pretrained baseline, runs full SFT, implements LoRA manually, and compares trainable parameter counts plus qualitative generations.

The first cell installs `evaluate` and `rouge_score` if needed. Training defaults are intentionally small:

```python
MAX_TRAIN_SAMPLES = 5000
MAX_TEST_SAMPLES = 400
FULL_SFT_EPOCHS = 1
LORA_EPOCHS = 1
```

## Assignment 4

`assignment_4_retrieval_augmented_generation.ipynb` is the PubMedQA RAG assignment. It downloads the real PubMedQA JSON, builds `questions` and `documents`, uses a Hugging Face LangChain LM, embeds and chunks the documents, creates a Chroma vector store, builds an LCEL RAG chain, and evaluates RAG against a no-context baseline.

The first cell installs the LangChain, Chroma, sentence-transformers, and Hugging Face dependencies if needed. Evaluation defaults to a small subset:

```python
EVAL_N = 30
RETRIEVAL_K = 1
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
RESET_VECTOR_STORE = False
```

Set `RESET_VECTOR_STORE = True` after changing chunking or embedding settings so Chroma rebuilds the index.

## NLTK

Assignment 1 uses `nltk.word_tokenize`, lowercases all tokens, builds the vocabulary from the training file only, and caps the vocabulary with `MAX_VOC_SIZE`.
