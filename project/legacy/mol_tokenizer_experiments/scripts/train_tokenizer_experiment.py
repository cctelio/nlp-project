#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mol_tokenizer_experiments.data import get_or_prepare_dataset, to_sft_messages
from mol_tokenizer_experiments.metrics import compute_generation_metrics
from mol_tokenizer_experiments.smiles_tokenizers import (
    add_strategy_tokens_to_tokenizer,
    assert_roundtrip,
    average_encoded_smiles_length,
    build_tokenizer_strategy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Mol-Instructions tokenizer strategy SFT and evaluation.")
    parser.add_argument("--tokenizer_strategy", default="default_bpe")
    parser.add_argument("--model_id", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--processed_dir", default=str(PROJECT_ROOT / "data" / "processed" / "molinstructions"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--force_prepare", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_eval_generations", type=int, default=512)
    parser.add_argument("--eval_generation_batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--wandb_project", default="MolInstructions-tokenizers")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_run_name(args):
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dry = "-dry" if args.dry_run else ""
    return f"molinstr-{args.tokenizer_strategy}-ep{args.num_train_epochs:g}-lr{args.learning_rate:g}{dry}-{timestamp}"


def resolve_output_dir(output_root, run_name):
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def compute_backend_name() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.device_count()}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_sft_dataset(dataset, strategy):
    return dataset.map(
        lambda batch: to_sft_messages(batch, strategy),
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Formatting SFT messages for {strategy.name}",
    )


def initialize_new_embeddings(model, old_vocab_size):
    embeddings = model.get_input_embeddings().weight.data
    if embeddings.shape[0] <= old_vocab_size:
        return
    mean_embedding = embeddings[:old_vocab_size].mean(dim=0, keepdim=True)
    embeddings[old_vocab_size:] = mean_embedding
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings.weight.shape[0] > old_vocab_size:
        output_embeddings.weight.data[old_vocab_size:] = output_embeddings.weight.data[:old_vocab_size].mean(
            dim=0,
            keepdim=True,
        )


def load_model_and_tokenizer(args, strategy):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_vocab_size = len(tokenizer)
    added_vocab_size = add_strategy_tokens_to_tokenizer(tokenizer, strategy)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    if added_vocab_size:
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))
        initialize_new_embeddings(model, old_vocab_size)
    return model, tokenizer, added_vocab_size


def generate_predictions(model, tokenizer, dataset, strategy, args):
    model.eval()
    device = next(model.parameters()).device
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rows = []
    n = min(args.num_eval_generations, len(dataset))
    subset = dataset.select(range(n))
    for start in range(0, n, args.eval_generation_batch_size):
        batch = subset.select(range(start, min(start + args.eval_generation_batch_size, n)))
        messages = [[{"role": "user", "content": instruction}] for instruction in batch["instruction"]]
        prompts = [
            tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for local_idx, output_ids in enumerate(generated):
            completion = tokenizer.decode(output_ids[prompt_width:], skip_special_tokens=True).strip()
            rows.append(
                {
                    "example_index": start + local_idx,
                    "instruction": batch["instruction"][local_idx],
                    "target_smiles": batch["target_smiles"][local_idx],
                    "generated_text": completion,
                    "predicted_smiles": strategy.decode_generated_text(completion),
                }
            )
    tokenizer.padding_side = original_padding_side
    return rows


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    is_main_process = global_rank == 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if args.dry_run:
        args.max_train_samples = args.max_train_samples or 64
        args.max_eval_samples = args.max_eval_samples or 32
        args.num_eval_generations = min(args.num_eval_generations, 32)
        args.num_train_epochs = min(args.num_train_epochs, 1)
        args.eval_steps = min(args.eval_steps, 10)
        args.save_steps = min(args.save_steps, 10)
    set_seed(args.seed)
    if not args.disable_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    run_name = build_run_name(args)
    output_dir = resolve_output_dir(args.output_dir, run_name)

    if is_main_process:
        print(f"Writing outputs to {output_dir}", flush=True)
        print(f"Tokenizer strategy: {args.tokenizer_strategy}", flush=True)
        print(f"Compute backend: {compute_backend_name()}", flush=True)

    dataset = get_or_prepare_dataset(
        args.processed_dir,
        seed=args.seed,
        test_size=args.test_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        force_prepare=args.force_prepare,
    )
    strategy = build_tokenizer_strategy(args.tokenizer_strategy)
    strategy.fit(dataset["train"]["target_smiles"])
    assert_roundtrip(strategy, dataset["train"]["target_smiles"])

    model, tokenizer, added_vocab_size = load_model_and_tokenizer(args, strategy)
    strategy_dir = output_dir / "tokenizer_strategy"
    strategy.save(strategy_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    train_dataset = prepare_sft_dataset(dataset["train"], strategy)
    eval_dataset = prepare_sft_dataset(dataset["test"], strategy)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none" if args.disable_wandb else "wandb",
        max_length=args.max_length,
        packing=torch.cuda.is_available(),
        dataset_kwargs={"skip_prepare_dataset": False, "dataset_num_proc": 4},
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_result = trainer.evaluate()
    if not is_main_process:
        return

    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(output_dir / "final_model")

    generation_model = trainer.accelerator.unwrap_model(trainer.model)
    records = generate_predictions(generation_model, tokenizer, dataset["test"], strategy, args)
    metrics, prediction_df = compute_generation_metrics(records)
    avg_train_len = average_encoded_smiles_length(tokenizer, strategy, dataset["train"]["target_smiles"])
    avg_eval_len = average_encoded_smiles_length(tokenizer, strategy, dataset["test"]["target_smiles"])
    metrics.update(
        {
            "tokenizer_strategy": strategy.name,
            "model_id": args.model_id,
            "added_vocab_size": int(added_vocab_size),
            "train_examples": int(len(dataset["train"])),
            "eval_examples": int(len(dataset["test"])),
            "avg_train_target_token_length": avg_train_len,
            "avg_eval_target_token_length": avg_eval_len,
            "train_loss": train_result.metrics.get("train_loss"),
            "eval_loss": eval_result.get("eval_loss"),
            "latest_checkpoint": get_last_checkpoint(str(output_dir)) or "",
        }
    )

    prediction_df.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    if not args.disable_wandb:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics)
            wandb.log({"predictions": wandb.Table(dataframe=prediction_df)})
            wandb.finish()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
