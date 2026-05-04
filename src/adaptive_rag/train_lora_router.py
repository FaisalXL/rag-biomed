"""Fine-tune Llama 3.1 8B as a sequence classifier with LoRA (PEFT) on MedHallu-derived CSV."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from adaptive_rag.config import DEFAULT_GENERATOR_MODEL, LORA_ROUTER_DIR, ensure_dirs


def _compute_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _move_model_to_training_device(model: torch.nn.Module) -> torch.nn.Module:
    """CUDA uses device_map at load time; MPS/CPU need explicit placement."""
    if torch.cuda.is_available():
        return model
    if torch.backends.mps.is_available():
        return model.to("mps")
    return model


def train_lora_router(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    model_id: str | None = None,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_train_steps: int | None = None,
) -> Path:
    ensure_dirs()
    model_id = model_id or DEFAULT_GENERATOR_MODEL
    output_dir = Path(output_dir or LORA_ROUTER_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, lineterminator="\n")
    except Exception:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        enc = tokenizer(examples["prompt"], truncation=True, max_length=2048)
        enc["labels"] = examples["label"]
        return enc

    _cols = dataset["train"].column_names
    tokenized = dataset.map(tokenize_fn, batched=True, num_proc=1, remove_columns=_cols)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    load_kw: dict = {
        "num_labels": 2,
        "dtype": _compute_dtype(),
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        load_kw["device_map"] = "cuda:0"
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **load_kw)
    model.config.pad_token_id = tokenizer.eos_token_id

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)
    model = _move_model_to_training_device(model)
    model.print_trainable_parameters()
    model.config.use_cache = False

    dev = next(model.parameters()).device
    print(f"Training on device: {dev} (MPS/CUDA avoids CPU-only runs that look 'stuck' on large models.)")

    base_ta = dict(
        output_dir=str(output_dir),
        learning_rate=2e-4,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=torch.cuda.is_available(),
        per_device_eval_batch_size=per_device_train_batch_size,
        weight_decay=0.01,
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    if max_train_steps is not None and max_train_steps > 0:
        print(f"max_train_steps={max_train_steps} (short run; skips per-epoch eval/save-best).")
        args = TrainingArguments(
            **base_ta,
            max_steps=max_train_steps,
            num_train_epochs=num_train_epochs,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
        )
        eval_ds = None
    else:
        args = TrainingArguments(
            **base_ta,
            num_train_epochs=num_train_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        eval_ds = tokenized["test"]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved adapter + tokenizer to {output_dir}")
    return output_dir
