# Berzelius No-Build Smoke Path

Use this path first when custom Apptainer builds are too slow or when project storage is near the file quota.

It follows the Berzelius Apptainer guide's simpler pattern:

```text
apptainer pull <image>.sif docker://pytorch/pytorch:<tag>
apptainer exec --nv <image>.sif ...
```

Instead of modifying the image, install this project's extra Python packages into a virtual environment stored in project storage.

## Assumptions

These commands use your current Berzelius values:

```text
Project storage: /proj/berzelius-2026-62
User:            x_telcr
Slurm account:  berzelius-2026-62
Repo:            /proj/berzelius-2026-62/users/x_telcr/nlp-project
```

Run commands from Berzelius.

## 1. Clean Any Interrupted Build

If a previous custom build was interrupted, remove its temporary directories. This can be slow if many files were created.

```bash
rm -rf /proj/berzelius-2026-62/users/x_telcr/apptainer_tmp
rm -rf /proj/berzelius-2026-62/users/x_telcr/apptainer_cache
```

## 2. Create Minimal Runtime Directories

```bash
mkdir -p /proj/berzelius-2026-62/users/x_telcr/containers
mkdir -p /proj/berzelius-2026-62/users/x_telcr/venvs
mkdir -p /proj/berzelius-2026-62/users/x_telcr/hf_cache
mkdir -p /proj/berzelius-2026-62/users/x_telcr/hf_datasets_cache
mkdir -p /proj/berzelius-2026-62/users/x_telcr/results
mkdir -p /proj/berzelius-2026-62/users/x_telcr/wandb
mkdir -p /proj/berzelius-2026-62/users/x_telcr/wandb_cache
```

## 3. Pull a PyTorch Image

This creates a `.sif` directly from Docker Hub. It should be much less painful than building a custom image with `%post`.

```bash
apptainer pull \
  /proj/berzelius-2026-62/users/x_telcr/containers/pytorch-2.4.1-cuda12.4.sif \
  docker://pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
```

## 4. Create a Project-Storage Virtualenv Inside the Image

Remove the old venv if it was created before this note. The first version could let `pip` install another `torch` inside the venv, which caused a Triton compiler failure during training.

```bash
rm -rf /proj/berzelius-2026-62/users/x_telcr/venvs/smollm
```

```bash
apptainer exec \
  --bind /proj/berzelius-2026-62/users/x_telcr:/work \
  /proj/berzelius-2026-62/users/x_telcr/containers/pytorch-2.4.1-cuda12.4.sif \
  bash -lc "python -m venv --system-site-packages /work/venvs/smollm && source /work/venvs/smollm/bin/activate && python -m pip install --upgrade pip setuptools wheel"
```

Install only what is needed for the smoke path, real-data preprocessing without RDKit canonicalization, and W&B. RDKit is intentionally omitted for now; install it later when running chemistry evaluation.

```bash
apptainer exec \
  --bind /proj/berzelius-2026-62/users/x_telcr:/work \
  /proj/berzelius-2026-62/users/x_telcr/containers/pytorch-2.4.1-cuda12.4.sif \
  bash -lc "source /work/venvs/smollm/bin/activate && python -m pip install \
    'datasets>=2.18.0' \
    'huggingface-hub>=0.23.0' \
    'pandas>=2.0.0' \
    'pyyaml>=6.0.0' \
    'selfies>=2.1.0' \
    'transformers>=4.40.0,<4.47.0' \
    'trl>=0.9.0,<0.13.0' \
    'accelerate>=0.30.0,<1.0.0' \
    'wandb>=0.16.0'"
```

Confirm that Python is using the container's PyTorch, not a venv-installed PyTorch:

```bash
apptainer exec \
  --bind /proj/berzelius-2026-62/users/x_telcr:/work \
  /proj/berzelius-2026-62/users/x_telcr/containers/pytorch-2.4.1-cuda12.4.sif \
  bash -lc "source /work/venvs/smollm/bin/activate && python -c 'import torch; print(torch.__version__); print(torch.__file__)'"
```

The path should point inside the container's Conda environment, not `/work/venvs/smollm/...`.

## 5. CPU Import Test

```bash
apptainer exec \
  --bind /proj/berzelius-2026-62/users/x_telcr:/work \
  /proj/berzelius-2026-62/users/x_telcr/containers/pytorch-2.4.1-cuda12.4.sif \
  bash -lc "source /work/venvs/smollm/bin/activate && python -c 'import torch, transformers, trl, datasets, pandas, yaml, selfies; print(torch.__version__); print(\"imports ok\")'"
```

## 6. Submit the No-Build GPU Smoke Job

The script uses tiny synthetic CSV splits and disables RDKit evaluation:

```text
slurm/berzelius_smoke_pulled_image.sh
```

Submit:

```bash
cd /proj/berzelius-2026-62/users/x_telcr/nlp-project/project
sbatch slurm/berzelius_smoke_pulled_image.sh
```

Monitor:

```bash
squeue -u x_telcr
tail -f slurm-smollm-pull-smoke-<jobid>.out
```

The script also writes a local GPU utilization log:

```text
/proj/berzelius-2026-62/users/x_telcr/results/gpu_logs/smollm-pull-smoke-<jobid>.csv
```

## 7. What This Tests

This test checks:

- Apptainer can run the PyTorch image.
- The project imports.
- Hugging Face model download works.
- TRL SFT training starts.
- The atomwise tokenizer path works.
- Checkpoints write to project storage.

It does not test:

- Mol-Instructions download.
- SELFIES decoding.
- RDKit evaluation.
- Real chemistry metrics.

Those should come after this infrastructure test passes.

## 8. Run a Short Real-Data Training Test

After the tiny smoke job passes, run:

```text
slurm/berzelius_short_train_pulled_image.sh
```

This job prepares 1024 Mol-Instructions examples and trains for 50 steps with evaluation generation disabled. It checks the real dataset preprocessing path and a longer training loop without spending full experiment compute.

Mol-Instructions stores the relevant molecular targets as SELFIES, so this short job requires the `selfies` package. It disables RDKit canonicalization for now so RDKit is not required yet.

Submit:

```bash
cd /proj/berzelius-2026-62/users/x_telcr/nlp-project/project
sbatch slurm/berzelius_short_train_pulled_image.sh
```

Monitor:

```bash
squeue -u x_telcr
tail -f slurm-smollm-pull-short-<jobid>.out
```

The script also writes a local GPU utilization log:

```text
/proj/berzelius-2026-62/users/x_telcr/results/gpu_logs/smollm-pull-short-<jobid>.csv
```

To test a different tokenizer without editing the script:

```bash
sbatch --export=ALL,TOKENIZER_CONFIG=configs/tokenizers/default.yaml slurm/berzelius_short_train_pulled_image.sh
sbatch --export=ALL,TOKENIZER_CONFIG=configs/tokenizers/smilespe.yaml slurm/berzelius_short_train_pulled_image.sh
```

You can pass runtime overrides through simple Slurm environment variables. Avoid passing a multi-argument `EXTRA_ARGS` string; Slurm/shell quoting can split it incorrectly.

```bash
sbatch --export=ALL,TOKENIZER_CONFIG=configs/tokenizers/atomwise.yaml,PER_DEVICE_TRAIN_BATCH_SIZE=8,GRADIENT_ACCUMULATION_STEPS=4,GRADIENT_CHECKPOINTING=false,PACKING=true,MAX_LENGTH=512 slurm/berzelius_short_train_pulled_image.sh
```

Packing is often useful for these experiments because many target strings are much shorter than the configured `training.max_length`. Benchmark it rather than assuming it is always faster, since packing changes the number and shape of training sequences.

## 9. Monitoring GPU Usage

Berzelius provides cluster tools and you can also log metrics yourself:

- `jobgraph -j <jobID>` generates a PNG with resource usage after or during a job.
- `jobsh -j <jobID>` opens a shell on the node running your job. Inside it, use `nvidia-smi` or `nvtop` for live GPU utilization.
- The pulled-image Slurm scripts in this repo write `nvidia-smi` samples every 10 seconds under `results/gpu_logs/`.

For live monitoring:

```bash
jobsh -j <jobID>
watch -n 2 nvidia-smi
```

For a Berzelius-generated graph:

```bash
jobgraph -j <jobID>
ls -lh *.png
```

For the CSV written by the script:

```bash
tail -n 20 /proj/berzelius-2026-62/users/x_telcr/results/gpu_logs/smollm-pull-short-<jobid>.csv
```

W&B can also record GPU utilization, but keep it disabled until the local GPU logs show the job is using the GPU reasonably.

## If You Saw `Failed to find C compiler`

That error came from Triton in a newer PyTorch stack installed inside the venv. Recreate the venv with `--system-site-packages` as shown above so it reuses the PyTorch bundled in the pulled image.

## Tiny RDKit Evaluation Job

After installing RDKit, submit a minimal evaluation-enabled run with:

```bash
bash slurm/submit_berzelius_eval_tiny.sh
```

This wrapper runs atomwise for 10 steps with:

```text
CANONICALIZE_SMILES=true
EVALUATION_RUN_AFTER_TRAINING=true
EVALUATION_MAX_VALIDATION_EXAMPLES=8
```

It is meant to test RDKit canonicalization, generation, and metrics before larger evaluation jobs.

## Instruct Model Test

The first quality-focused test should use the model's chat template and TRL's prompt-completion dataset shape. This repository now supports that through:

```text
SFT_FORMAT=prompt_completion
USE_CHAT_TEMPLATE=true
MODEL_ID=HuggingFaceTB/SmolLM-135M-Instruct
```

The ready-to-run command is stored in:

```text
slurm/sbatch_instruct_atomwise_500_eval.txt
```

Run it with:

```bash
bash slurm/sbatch_instruct_atomwise_500_eval.txt
```

This runs atomwise for 500 steps, uses completion-only loss when the installed TRL version supports it, and evaluates 32 validation examples.
