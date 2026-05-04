"""Train lightweight routers: XGBoost on MiniLM embeddings, DistilBERT classifier."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from adaptive_rag.config import DEFAULT_EMBEDDER_MODEL, DISTILBERT_ROUTER_DIR, ensure_dirs


def train_xgboost_router(
    prompts: list,
    labels: list,
    embedder_id: str | None = None,
    n_estimators: int = 200,
    max_depth: int = 6,
) -> tuple[xgb.XGBClassifier, SentenceTransformer]:
    embedder_id = embedder_id or DEFAULT_EMBEDDER_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(embedder_id, device=device)
    X_train, X_test, y_train, y_test = train_test_split(
        prompts, labels, test_size=0.1, random_state=42
    )
    Xtr = embedder.encode(X_train, batch_size=256, convert_to_numpy=True)
    Xte = embedder.encode(X_test, batch_size=256, convert_to_numpy=True)

    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_metric="logloss",
    )
    clf.fit(Xtr, y_train)
    print(f"XGBoost fit in {time.time() - t0:.2f}s; acc={accuracy_score(y_test, clf.predict(Xte)):.4f}")
    return clf, embedder


def train_distilbert_router(
    prompts: list,
    labels: list,
    output_dir: str | Path | None = None,
    num_train_epochs: int = 3,
) -> Path:
    ensure_dirs()
    output_dir = Path(output_dir or DISTILBERT_ROUTER_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        prompts, labels, test_size=0.1, random_state=42
    )
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    test_ds = Dataset.from_dict({"text": X_test, "label": y_test})

    def tokenize_function(examples):
        return tok(examples["text"], padding="max_length", truncation=True, max_length=512)

    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    def compute_metrics(eval_pred):
        logits, labs = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labs).mean())}

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"Saved DistilBERT router to {output_dir}")
    return output_dir
