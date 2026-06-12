#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig as TrlSFTConfig, SFTTrainer

from src.config import (
    MODELS,
    SFT_DEFAULTS,
    get_model_id,
    local_model_path,
    resolve_model_source,
    adapter_path,
)
from src.data_utils import load_split
from src.prompt_template import build_sft_messages


def _build_label_json(label: int, cwe_id: str = "") -> Dict[str, Any]:
    if label == 1:
        result: Dict[str, Any] = {
            "vulnerable": True,
            "explanation": "This code contains a security vulnerability.",
        }
        if cwe_id:
            result["cwe_id"] = cwe_id
        return result
    return {"vulnerable": False}


def format_dataset_for_sft(ds: Dataset, tokenizer: AutoTokenizer, max_chars: int = 0) -> Dataset:
    formatted = []
    skipped = 0
    for row in ds:
        if max_chars and len(row["text"]) > max_chars:
            skipped += 1
            continue
        label_json = _build_label_json(row["label"], row.get("cwe_id", ""))
        messages = build_sft_messages(row["text"], label_json)

        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = ""
            for msg in messages:
                text += f"### {msg['role'].capitalize()}\n{msg['content']}\n\n"

        formatted.append({"text": text})

    if skipped:
        print(f"[sft_train] Dropped {skipped} examples exceeding max_chars={max_chars}")
    return Dataset.from_list(formatted)


def train(
    model_id: str,
    max_seq_length: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Path:
    cfg = SFT_DEFAULTS
    full_model_id = get_model_id(model_id)
    output_dir = adapter_path(full_model_id)
    max_seq_length = max_seq_length or cfg.max_seq_length

    resolved = resolve_model_source(full_model_id)
    is_local = resolved != full_model_id
    print(f"[sft_train] Model     : {full_model_id}"
          f"{' (local)' if is_local else ' (HuggingFace)'}")
    print(f"[sft_train] Output    : {output_dir}")
    print(f"[sft_train] Max seq   : {max_seq_length}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved, trust_remote_code=True, model_max_length=max_seq_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    if not is_local:
        local_dir = local_model_path(full_model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(local_dir))
        tokenizer.save_pretrained(str(local_dir))
        print(f"[sft_train] Saved base model locally → {local_dir}")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print("[sft_train] Loading datasets …")
    train_ds = load_split("train")
    val_ds = load_split("val")
    print(f"[sft_train] Train: {len(train_ds)},  Val: {len(val_ds)}")

    train_formatted = format_dataset_for_sft(train_ds, tokenizer, max_chars=cfg.max_train_chars)
    val_formatted = format_dataset_for_sft(val_ds, tokenizer, max_chars=cfg.max_train_chars)

    training_args = TrlSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optimizer,
        bf16=True,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=cfg.report_to,
        seed=cfg.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_length=max_seq_length,
        dataset_text_field="text",
        completion_only_loss=True,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_formatted,
        eval_dataset=val_formatted,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    print("[sft_train] Starting training …")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[sft_train] Final adapter saved to: {final_dir}")

    summary = {
        "model": full_model_id,
        "adapter_path": str(final_dir),
        "hyperparameters": {
            "learning_rate": cfg.learning_rate, "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
            "num_train_epochs": cfg.num_train_epochs,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "effective_batch_size": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
            "max_seq_length": max_seq_length, "max_train_chars": cfg.max_train_chars,
            "completion_only_loss": True, "optimizer": cfg.optimizer,
            "weight_decay": cfg.weight_decay, "warmup_ratio": cfg.warmup_ratio,
            "lr_scheduler": cfg.lr_scheduler_type,
            "quantization": f"{cfg.quantization_bits}-bit QLoRA",
            "precision": cfg.compute_dtype, "seed": cfg.seed,
        },
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: QLoRA fine-tuning")
    parser.add_argument("--model", required=True, help=f"Model short name or HF ID. Available: {list(MODELS.keys())}")
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train(model_id=args.model, max_seq_length=args.max_seq_length, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
