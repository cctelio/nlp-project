"""Text generation helpers for natural-language-to-SMILES evaluation."""

from __future__ import annotations

from typing import Any

from src.data.preprocessing import format_generation_prompt
from src.utils.reproducibility import set_global_seed


def _first_response_segment(text: str) -> str:
    """Keep only the first generated response segment."""
    if not text:
        return ""
    stripped = text.strip()
    for marker in ("\n###", "\nInstruction:", "\nResponse:"):
        if marker in stripped:
            stripped = stripped.split(marker, 1)[0].strip()
    return stripped


def generate_for_rows(
    model: Any,
    tokenizer: Any,
    representation: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    seed: int,
    prompt_template: str,
) -> list[dict[str, Any]]:
    """Generate raw and decoded SMILES strings for evaluation rows."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for generation.") from exc

    set_global_seed(seed)
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    prompts = [format_generation_prompt(str(row["instruction"]), template=prompt_template) for row in rows]
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if do_sample:
                generation_kwargs.update({"temperature": temperature, "top_p": top_p})
            outputs = model.generate(**inputs, **generation_kwargs)
            decoded = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
            for row, generated_text in zip(batch_rows, decoded, strict=False):
                response_text = _first_response_segment(generated_text)
                generated_smiles = representation.decode_generation(response_text)
                records.append(
                    {
                        "id": row.get("id", ""),
                        "instruction": row.get("instruction", ""),
                        "target_smiles": row.get("target_smiles", ""),
                        "generated_text": response_text,
                        "generated_smiles": generated_smiles,
                    }
                )
    return records
