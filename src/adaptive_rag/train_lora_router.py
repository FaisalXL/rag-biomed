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


def train_lora_router(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    model_id: str | None = None,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
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
        return tokenizer(examples["prompt"], truncation=True, max_length=2048)

    tokenized = dataset.map(tokenize_fn, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    load_kw: dict = {
        "num_labels": 2,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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
    model.print_trainable_parameters()
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-4,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=torch.cuda.is_available(),
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=25,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved adapter + tokenizer to {output_dir}")
    return output_dir
