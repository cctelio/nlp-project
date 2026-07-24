# Running on Berzelius with Apptainer

This guide is for the main project under `project/`.

NSC documentation:

- Berzelius getting started: <https://www.nsc.liu.se/support/systems/berzelius-getting-started/>
- Berzelius compute node guide: <https://www.nsc.liu.se/support/systems/berzelius-gpu/>
- Berzelius Apptainer guide: <https://www.nsc.liu.se/support/systems/berzelius-software/berzelius-apptainer/>

## Why Berzelius Is a Good First Target

Berzelius is designed for AI/ML work and has NVIDIA A100 and H200 GPU nodes. It also supports Apptainer directly. This avoids the ARM64 container issue encountered on Arrhenius Grace Hopper nodes when an image is only built for AMD64.

For this project, start on Berzelius Ampere with one A100 GPU. That is enough for smoke tests and small SmolLM-135M runs.

## Storage Layout

Do not store containers, Hugging Face caches, W&B logs, datasets, or results in `/home`. Berzelius home has a small quota.

Use project storage:

```text
/proj/<berzelius_project>/users/telio/
```

Suggested layout:

```text
/proj/<berzelius_project>/users/telio/
  nlp-project/              # Git clone
    project/                # This SmolLM project
  containers/
  hf_cache/
  hf_datasets_cache/
  wandb/
  wandb_cache/
  results/
  apptainer_cache/
  apptainer_tmp/
```

Find your project/account names:

```bash
projinfo
nscquota
ls /proj
```

## Clone or Copy the Repository

From Berzelius:

```bash
ssh telio@berzelius.nsc.liu.se
```

Use the project directory you actually have access to:

```bash
mkdir -p /proj/<berzelius_project>/users/telio
cd /proj/<berzelius_project>/users/telio
git clone <your-github-repo-url> nlp-project
cd nlp-project/project
```

Create working directories:

```bash
mkdir -p /proj/<berzelius_project>/users/telio/containers
mkdir -p /proj/<berzelius_project>/users/telio/hf_cache
mkdir -p /proj/<berzelius_project>/users/telio/hf_datasets_cache
mkdir -p /proj/<berzelius_project>/users/telio/wandb
mkdir -p /proj/<berzelius_project>/users/telio/wandb_cache
mkdir -p /proj/<berzelius_project>/users/telio/results
mkdir -p /proj/<berzelius_project>/users/telio/apptainer_cache
mkdir -p /proj/<berzelius_project>/users/telio/apptainer_tmp
```

## Build the Container

You do not need a GPU just to build the container. The build downloads the base image and installs Python packages; it does not run training or use CUDA devices. You only need a GPU later to test `torch.cuda.is_available()` and to train.

If login-node builds are allowed and the machine is not busy, you can build directly after logging in. If you prefer not to build on the login node, request a CPU/regular interactive allocation. A GPU allocation is optional for building.

Optional interactive allocation:

```bash
interactive -A <berzelius_project_account> -t 00-02:00:00
```

Build with fakeroot into project storage.

Use the Berzelius-specific recipe first. It uses the standard PyTorch CUDA runtime image, which is smaller than the broad NVIDIA NGC image used for Arrhenius compatibility.

```bash
cd /proj/<berzelius_project>/users/telio/nlp-project/project

APPTAINER_CACHEDIR=/proj/<berzelius_project>/users/telio/apptainer_cache \
APPTAINER_TMPDIR=/proj/<berzelius_project>/users/telio/apptainer_tmp \
TMPDIR=/proj/<berzelius_project>/users/telio/apptainer_tmp \
apptainer build --fakeroot \
  /proj/<berzelius_project>/users/telio/containers/smollm-chemical-tokenization.sif \
  containers/smollm-chemical-tokenization-berzelius.def
```

If this recipe fails because the base image is not available on the node architecture, fall back to:

```text
containers/smollm-chemical-tokenization.def
```

That fallback is larger because it uses NVIDIA's NGC PyTorch image.

Test imports and GPU access:

```bash
apptainer exec --nv \
  /proj/<berzelius_project>/users/telio/containers/smollm-chemical-tokenization.sif \
  python -c "import torch, transformers, trl, rdkit, selfies; print(torch.__version__); print(torch.cuda.is_available())"
```

The final printed value should be:

```text
True
```

If you do not have a GPU allocation yet, run a CPU-only import test instead:

```bash
apptainer exec \
  /proj/<berzelius_project>/users/telio/containers/smollm-chemical-tokenization.sif \
  python -c "import torch, transformers, trl, rdkit, selfies; print(torch.__version__); print('container imports ok')"
```

After a successful build, remove build caches. Your project has a tight file quota, and interrupted Apptainer builds can leave many files behind:

```bash
rm -rf /proj/<berzelius_project>/users/telio/apptainer_cache
rm -rf /proj/<berzelius_project>/users/telio/apptainer_tmp
```

## Run the Cheapest Full-Pipeline Smoke Test

Edit:

```text
slurm/berzelius_smoke_container.sh
```

Set:

```bash
#SBATCH -A <berzelius_project_account>
PROJECT_BASE="/proj/<berzelius_project>/users/telio"
```

Submit from `project/`:

```bash
sbatch slurm/berzelius_smoke_container.sh
```

Monitor:

```bash
squeue -u telio
tail -f slurm-smollm-smoke-<jobid>.out
```

This job:

1. Downloads/prepares only 128 examples.
2. Runs the `atomwise` tokenizer strategy.
3. Trains for 2 steps.
4. Evaluates only 4 validation examples.
5. Writes outputs under:

```text
/proj/<berzelius_project>/users/telio/results/smoke_runs/
```

## Run a Small Short Training Job

Only after the smoke test works, edit:

```text
slurm/berzelius_short_train_container.sh
```

Set the same project account and `PROJECT_BASE`.

Submit:

```bash
sbatch slurm/berzelius_short_train_container.sh
```

This job uses 1024 examples and caps training at 50 steps. It is still a test, not the final experiment.

To test another tokenizer, edit:

```bash
TOKENIZER_CONFIG="configs/tokenizers/default.yaml"
```

or:

```bash
TOKENIZER_CONFIG="configs/tokenizers/smilespe.yaml"
```

## W&B

Keep W&B disabled for the smoke test.

When ready, add these to the Slurm script before `apptainer exec`:

```bash
export WANDB_API_KEY="<your_api_key>"
export WANDB_PROJECT="SmolLM-chemical-tokenization"
```

Then change the training command override from:

```bash
--set logging.disable_wandb=true
```

to:

```bash
--set logging.disable_wandb=false
```

Do not commit your API key.

## Final Experiment Order

Do not run HPO first.

Use this order:

1. Build and test the container.
2. Run `berzelius_smoke_container.sh`.
3. Run `berzelius_short_train_container.sh` with `atomwise`.
4. Repeat the short job for `default` and `smilespe`.
5. Increase steps/epochs only after all three strategies run.
6. Run HPO only when the basic comparison works.
