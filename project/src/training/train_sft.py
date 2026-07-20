"""Supervised fine-tuning workflow for tokenizer comparison experiments."""

from __future__ import annotations

import inspect
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.data.preprocessing import PROMPT_TEMPLATE, format_sft_text
from src.data.sft import precompute_sft_splits
from src.evaluation.evaluate import evaluate_rows
from src.tokenization.base import load_representation
from src.tokenization.registry import build_tokenizer, build_tokenizer_for_representation, save_tokenizer_artifacts
from src.utils.config import dump_yaml
from src.utils.logging import read_rows, setup_logging, write_json
from src.utils.paths import resolve_path, resolve_run_dir
from src.utils.reproducibility import set_global_seed, stable_int_seed


def _require_training_dependencies():
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM
        from transformers.trainer_utils import get_last_checkpoint
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Training requires torch, datasets, transformers, and trl. "
            "Install project dependencies before running training."
        ) from exc
    return torch, Dataset, AutoModelForCausalLM, get_last_checkpoint, SFTConfig, SFTTrainer


def _signature_has(callable_obj: Any, key: str) -> bool:
    return key in inspect.signature(callable_obj).parameters


def _set_strategy_kwargs(target_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs by constructor signature."""
    parameters = inspect.signature(target_cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def _resolve_torch_dtype(torch: Any, dtype_name: str):
    if dtype_name in {"auto", "", None}:
        return None
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _prepare_text_dataset(Dataset: Any, rows: list[dict[str, Any]], representation: Any, tokenizer: Any, prompt_template: str):
    eos_token = tokenizer.eos_token or ""
    formatted_rows = []
    for row in rows:
        if row.get("text"):
            formatted_rows.append(dict(row))
            continue
        response = row.get("response_smiles") or representation.encode_smiles(str(row["target_smiles"]))
        formatted_rows.append(
            {
                **row,
                "text": format_sft_text(str(row["instruction"]), response=response, eos_token=eos_token, template=prompt_template),
            }
        )
    return Dataset.from_list(formatted_rows)


def _make_sft_config(SFTConfig: Any, run_dir: Path, config: dict[str, Any], report_to: str | list[str]):
    training = config.get("training", {})
    bf16 = bool(training.get("bf16", True))
    if bf16:
        try:
            import torch

            bf16 = torch.cuda.is_available() and (
                not hasattr(torch.cuda, "is_bf16_supported") or torch.cuda.is_bf16_supported()
            )
        except Exception:
            bf16 = False
    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir),
        "num_train_epochs": training.get("num_train_epochs", 3),
        "max_steps": training.get("max_steps", -1),
        "per_device_train_batch_size": training.get("per_device_train_batch_size", 4),
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 8),
        "learning_rate": training.get("learning_rate", 2e-5),
        "warmup_ratio": training.get("warmup_ratio", 0.05),
        "weight_decay": training.get("weight_decay", 0.0),
        "logging_steps": training.get("logging_steps", 10),
        "save_steps": training.get("save_steps", 250),
        "save_total_limit": training.get("save_total_limit", 2),
        "load_best_model_at_end": training.get("load_best_model_at_end", True),
        "metric_for_best_model": training.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": training.get("greater_is_better", False),
        "report_to": report_to,
        "run_name": training.get("run_name"),
        "bf16": bf16,
        "fp16": training.get("fp16", False),
        "packing": training.get("packing", False),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "dataloader_num_workers": training.get("dataloader_num_workers", 0),
        "max_grad_norm": training.get("max_grad_norm", 1.0),
        "dataset_text_field": "text",
    }
    strategy_key = "eval_strategy" if _signature_has(SFTConfig.__init__, "eval_strategy") else "evaluation_strategy"
    kwargs[strategy_key] = training.get("eval_strategy", "steps")
    kwargs["eval_steps"] = training.get("eval_steps", 250)
    kwargs["save_strategy"] = training.get("save_strategy", "steps")

    if _signature_has(SFTConfig.__init__, "max_length"):
        kwargs["max_length"] = training.get("max_length", 512)
    elif _signature_has(SFTConfig.__init__, "max_seq_length"):
        kwargs["max_seq_length"] = training.get("max_length", 512)

    return SFTConfig(**_set_strategy_kwargs(SFTConfig, kwargs))


def _strategy_sft_paths(data_config: dict[str, Any], strategy: str) -> dict[str, str]:
    sft_dir = resolve_path(data_config.get("sft_dir", "results/data/sft")) / strategy
    extension = str(data_config.get("sft_file_format", "csv")).lstrip(".")
    return {
        "train_path": str(sft_dir / f"train.{extension}"),
        "validation_path": str(sft_dir / f"validation.{extension}"),
        "test_path": str(sft_dir / f"test.{extension}"),
        "representation_path": str(sft_dir / "smiles_representation.json"),
    }


def _resolve_dataset_paths(data_config: dict[str, Any], strategy: str) -> tuple[dict[str, str], bool]:
    explicit_sft_train = data_config.get("sft_train_path")
    explicit_sft_validation = data_config.get("sft_validation_path")
    if data_config.get("use_precomputed_sft", False) and explicit_sft_train and explicit_sft_validation:
        return (
            {
                "train_path": str(resolve_path(explicit_sft_train)),
                "validation_path": str(resolve_path(explicit_sft_validation)),
                "test_path": str(resolve_path(data_config.get("sft_test_path"))) if data_config.get("sft_test_path") else "",
                "representation_path": str(resolve_path(data_config.get("representation_path")))
                if data_config.get("representation_path")
                else "",
            },
            True,
        )

    default_sft = _strategy_sft_paths(data_config, strategy)
    if (
        data_config.get("use_precomputed_sft", False)
        and Path(default_sft["train_path"]).exists()
        and Path(default_sft["validation_path"]).exists()
    ):
        return default_sft, True

    return (
        {
            "train_path": str(resolve_path(data_config.get("train_path", "results/data/train.jsonl"))),
            "validation_path": str(resolve_path(data_config.get("validation_path", "results/data/validation.jsonl"))),
            "test_path": str(resolve_path(data_config.get("test_path"))) if data_config.get("test_path") else "",
            "representation_path": "",
        },
        False,
    )


def materialize_sft_data_for_config(config: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Precompute SFT JSONL files for a config and return paths usable by training."""
    data_config = config.get("data", {})
    paths = precompute_sft_splits(
        train_path=resolve_path(data_config.get("train_path", "results/data/train.jsonl")),
        validation_path=resolve_path(data_config.get("validation_path", "results/data/validation.jsonl")),
        test_path=resolve_path(data_config.get("test_path")) if data_config.get("test_path") else None,
        output_dir=output_dir,
        tokenizer_config=config.get("tokenizer", {}),
        prompt_template=data_config.get("prompt_template") or PROMPT_TEMPLATE,
    )
    if str(data_config.get("sft_file_format", "csv")).lstrip(".") == "csv":
        return {
            "train_path": paths["train_csv_path"],
            "validation_path": paths["validation_csv_path"],
            "test_path": paths["test_csv_path"],
            "representation_path": paths["representation_path"],
            "manifest_path": paths["manifest_path"],
        }
    return paths


def train_from_config(config: dict[str, Any]) -> Path:
    """Run one SFT experiment and return the run directory."""
    torch, Dataset, AutoModelForCausalLM, get_last_checkpoint, SFTConfig, SFTTrainer = _require_training_dependencies()

    config = deepcopy(config)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    tokenizer_config = config.get("tokenizer", {})
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})
    if config.get("dry_run", False):
        training_config["max_steps"] = int(training_config.get("max_steps", 2))
        if training_config["max_steps"] < 0:
            training_config["max_steps"] = 2
        training_config["eval_steps"] = max(1, min(int(training_config.get("eval_steps", 1)), 1))
        training_config["save_steps"] = max(1, min(int(training_config.get("save_steps", 1)), 1))
        training_config["logging_steps"] = max(1, min(int(training_config.get("logging_steps", 1)), 1))
        training_config["save_total_limit"] = max(1, min(int(training_config.get("save_total_limit", 1)), 1))
        config.setdefault("evaluation", {})
        config["evaluation"]["max_validation_examples"] = min(
            int(config["evaluation"].get("max_validation_examples", 4)),
            int(config.get("dry_run_eval_samples", 16)),
        )
    if training_config.get("bf16", False):
        bf16_supported = torch.cuda.is_available() and (
            not hasattr(torch.cuda, "is_bf16_supported") or torch.cuda.is_bf16_supported()
        )
        if not bf16_supported:
            training_config["bf16"] = False
    prompt_template = data_config.get("prompt_template") or PROMPT_TEMPLATE

    strategy = tokenizer_config.get("strategy", "default")
    output_root = resolve_path(training_config.get("output_root", "results/runs"))
    run_dir = resolve_run_dir(output_root, strategy=strategy, run_name=training_config.get("run_name"))
    logger = setup_logging(run_dir / "train.log")
    dump_yaml(run_dir / "config.yaml", config)

    dataset_paths, using_precomputed_sft = _resolve_dataset_paths(data_config, strategy)
    train_rows = read_rows(dataset_paths["train_path"])
    validation_rows = read_rows(dataset_paths["validation_path"])
    test_rows = read_rows(dataset_paths["test_path"]) if dataset_paths.get("test_path") else []

    if config.get("dry_run", False):
        train_rows = train_rows[: int(config.get("dry_run_train_samples", 32))]
        validation_rows = validation_rows[: int(config.get("dry_run_eval_samples", 16))]
        test_rows = test_rows[: int(config.get("dry_run_eval_samples", 16))]

    split_smiles = {
        "train": [str(row["target_smiles"]) for row in train_rows],
        "validation": [str(row["target_smiles"]) for row in validation_rows],
    }
    if test_rows:
        split_smiles["test"] = [str(row["target_smiles"]) for row in test_rows]

    representation_path = dataset_paths.get("representation_path")
    if using_precomputed_sft and representation_path and Path(representation_path).exists():
        representation = load_representation(representation_path)
        tokenizer_result = build_tokenizer_for_representation(
            model_id=model_config.get("model_id", "HuggingFaceTB/SmolLM-135M"),
            representation=representation,
            train_smiles=split_smiles["train"],
            split_smiles=split_smiles,
            tokenizer_revision=model_config.get("revision"),
        )
    else:
        tokenizer_result = build_tokenizer(
            model_id=model_config.get("model_id", "HuggingFaceTB/SmolLM-135M"),
            strategy_config=tokenizer_config,
            train_smiles=split_smiles["train"],
            split_smiles=split_smiles,
            tokenizer_revision=model_config.get("revision"),
        )
    save_tokenizer_artifacts(tokenizer_result, run_dir)
    write_json(run_dir / "tokenizer_metrics.json", tokenizer_result.metrics)
    logger.info("Tokenizer metrics: %s", tokenizer_result.metrics)

    train_dataset = _prepare_text_dataset(Dataset, train_rows, tokenizer_result.representation, tokenizer_result.tokenizer, prompt_template)
    validation_dataset = _prepare_text_dataset(
        Dataset,
        validation_rows,
        tokenizer_result.representation,
        tokenizer_result.tokenizer,
        prompt_template,
    )

    dtype = _resolve_torch_dtype(torch, model_config.get("torch_dtype", "bfloat16"))
    model_kwargs: dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    attn_implementation = model_config.get("attn_implementation", "auto")
    if attn_implementation != "auto":
        model_kwargs["_attn_implementation"] = attn_implementation
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available() and model_config.get("device_map", "local_rank") == "local_rank":
        torch.cuda.set_device(local_rank)
        model_kwargs["device_map"] = {"": local_rank}

    logger.info("Loading model %s", model_config.get("model_id", "HuggingFaceTB/SmolLM-135M"))
    model = AutoModelForCausalLM.from_pretrained(
        model_config.get("model_id", "HuggingFaceTB/SmolLM-135M"),
        revision=model_config.get("revision"),
        **model_kwargs,
    )
    model.resize_token_embeddings(len(tokenizer_result.tokenizer))

    if not logging_config.get("disable_wandb", True):
        os.environ["WANDB_PROJECT"] = logging_config.get("wandb_project", "SmolLM-chemical-tokenization")
        if training_config.get("run_name"):
            os.environ["WANDB_NAME"] = training_config["run_name"]
        report_to: str | list[str] = "wandb"
    else:
        report_to = "none"

    sft_config = _make_sft_config(SFTConfig, run_dir, config, report_to=report_to)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
    }
    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer_result.tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer_result.tokenizer
    if "dataset_text_field" in trainer_signature:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)
    logger.info("Starting training: train=%d validation=%d", len(train_rows), len(validation_rows))
    train_result = trainer.train(resume_from_checkpoint=training_config.get("resume_from_checkpoint"))
    trainer_eval = trainer.evaluate()

    final_model_dir = run_dir / "final_model"
    trainer.save_model(final_model_dir)
    tokenizer_result.tokenizer.save_pretrained(final_model_dir)
    tokenizer_result.representation.save(final_model_dir / "smiles_representation.json")

    train_metrics = {
        "train_loss": train_result.metrics.get("train_loss") if hasattr(train_result, "metrics") else None,
        "eval_loss": trainer_eval.get("eval_loss"),
        "latest_checkpoint": get_last_checkpoint(str(run_dir)) or "",
    }
    train_metrics.update(tokenizer_result.metrics)
    write_json(run_dir / "train_metrics.json", train_metrics)

    if not logging_config.get("disable_wandb", True):
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({f"tokenizer/{key}": value for key, value in tokenizer_result.metrics.items()})
                wandb.log({f"final/{key}": value for key, value in train_metrics.items() if value is not None})
                for key, value in train_metrics.items():
                    if value is not None:
                        wandb.run.summary[key] = value
        except Exception as exc:
            logger.warning("Could not log final metrics to W&B: %s", exc)

    if config.get("evaluation", {}).get("run_after_training", True):
        generation_model = trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, "accelerator") else trainer.model
        eval_config = dict(config.get("generation", {}))
        eval_config["seed"] = stable_int_seed(seed, "validation-generation")
        eval_rows = validation_rows[: int(config.get("evaluation", {}).get("max_validation_examples", len(validation_rows)))]
        validation_metrics = evaluate_rows(
            model=generation_model,
            tokenizer=tokenizer_result.tokenizer,
            representation=tokenizer_result.representation,
            rows=eval_rows,
            split_name="validation",
            output_dir=run_dir / "evaluation",
            generation_config=eval_config,
            prompt_template=prompt_template,
        )
        combined_metrics = {**train_metrics, **validation_metrics}
        write_json(run_dir / "metrics.json", combined_metrics)

    logger.info("Finished run at %s", run_dir)
    return run_dir
