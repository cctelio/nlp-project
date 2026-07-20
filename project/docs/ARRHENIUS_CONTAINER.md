# Running on Arrhenius with Apptainer

This guide is for the main NLP project in the repository root, not the assignment notebooks.

Arrhenius documentation:

- Quick start: <https://hpc.pages.naiss.se/user-documentation/support-docs/arrhenius_hpc/quickstart/>
- Containers: <https://hpc.pages.naiss.se/user-documentation/support-docs/arrhenius_hpc/software_development/containers/>

## Key Arrhenius Rules

Arrhenius uses Apptainer for containers. The documentation says container image files should not be stored in `$HOME`; images should normally live in project storage, for example under:

```text
/nobackup/proj/disk/<project>/
```

It also says to build the container on the same node type where it will run. For this project, build on a GPU node if you plan to train on the GPU partition.

## Suggested Project Layout on Arrhenius

Replace `<project>` and `<username>` with your real allocation/project and login name.

```text
/nobackup/proj/disk/<project>/telio/
  nlp-project/              # Git clone of this repository
    project/                # Main SmolLM project code
  containers/
    smollm-chemical-tokenization.sif
  hf_cache/
  hf_datasets_cache/
  wandb/
  results/
```

## Clone the Repository

```bash
ssh <username>@login.hpc.arrhenius.naiss.se
cd /nobackup/proj/disk/<project>/telio
git clone <your-github-repo-url> nlp-project
cd nlp-project/project
```

## Build the Container

Start an interactive GPU allocation:

```bash
interactive -A <project> -p gpu --gpus=1 -t 02:00:00
```

Build the image from the recipe in this repository:

```bash
cd /nobackup/proj/disk/<project>/telio/nlp-project/project
mkdir -p /nobackup/proj/disk/<project>/telio/containers

apptainer build \
  /nobackup/proj/disk/<project>/telio/containers/smollm-chemical-tokenization.sif \
  containers/smollm-chemical-tokenization.def
```

Quick import test:

```bash
apptainer exec --nv \
  /nobackup/proj/disk/<project>/telio/containers/smollm-chemical-tokenization.sif \
  python -c "import torch, transformers, trl, rdkit; print(torch.__version__); print(torch.cuda.is_available())"
```

## Prepare Data

Run preprocessing once. It downloads Mol-Instructions from Hugging Face and writes canonical CSV splits.

```bash
apptainer exec --nv \
  --bind /nobackup/proj/disk/<project>/telio:/work \
  /nobackup/proj/disk/<project>/telio/containers/smollm-chemical-tokenization.sif \
  bash -lc "cd /work/nlp-project/project && python scripts/prepare_data.py \
    --config configs/default.yaml \
    --set data.cache_dir=/work/hf_cache/mol_instructions \
    --set data.processed_dir=/work/results/data/mol_instructions_description_guided \
    --set data.train_path=/work/results/data/mol_instructions_description_guided/train.csv \
    --set data.validation_path=/work/results/data/mol_instructions_description_guided/validation.csv \
    --set data.test_path=/work/results/data/mol_instructions_description_guided/test.csv"
```

## Submit a Training Job

From the `project/` directory, edit `slurm/arrhenius_train_container.sh` and set:

```bash
#SBATCH -A <project>
PROJECT_BASE="/nobackup/proj/disk/<project>/telio"
```

Then submit:

```bash
sbatch slurm/arrhenius_train_container.sh
```

To run a different tokenizer strategy:

```bash
sbatch --export=ALL,TOKENIZER_CONFIG=configs/tokenizers/default.yaml slurm/arrhenius_train_container.sh
sbatch --export=ALL,TOKENIZER_CONFIG=configs/tokenizers/smilespe.yaml slurm/arrhenius_train_container.sh
```

## Evaluate a Run

After training finishes, find the run directory in:

```text
/nobackup/proj/disk/<project>/telio/results/runs/
```

Then run:

```bash
apptainer exec --nv \
  --bind /nobackup/proj/disk/<project>/telio:/work \
  /nobackup/proj/disk/<project>/telio/containers/smollm-chemical-tokenization.sif \
  bash -lc "cd /work/nlp-project/project && python scripts/evaluate.py \
    --run_dir /work/results/runs/<run-name> \
    --split test"
```

## Notes

- Use project storage for data, outputs, Hugging Face cache, W&B cache, and the `.sif` image.
- Keep `$HOME` for small config files only.
- This project is single-node fine-tuning. Multi-node container work on Arrhenius needs extra care around the Slingshot interconnect, according to the Arrhenius container documentation.
- If a build fails, the Arrhenius docs suggest debugging with an Apptainer sandbox on local disk such as `/tmp`, then building the final `.sif` on project storage.
