"""End-to-end adaptive RAG benchmark: vanilla LLM, always-RAG, and router-conditioned paths."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from adaptive_rag.config import (
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_GENERATOR_MODEL,
    LORA_ROUTER_DIR,
    PUBMED_FAISS_INDEX,
    PUBMED_FAISS_MAPPING,
)


def load_generator(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    gen = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    return tok, gen


def load_lora_router(model_id: str, adapter_dir: str | Path):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    router = PeftModel.from_pretrained(base, str(adapter_dir))
    router.eval()
    return tok, router


def load_distilbert_router(router_dir: str | Path):
    router_dir = Path(router_dir)
    tok = AutoTokenizer.from_pretrained(str(router_dir))
    load_kw: dict = {}
    if torch.cuda.is_available():
        load_kw["device_map"] = "cuda:0"
    m = AutoModelForSequenceClassification.from_pretrained(str(router_dir), **load_kw)
    m.eval()
    return tok, m


def run_shootout(
    faiss_index_path: str | Path,
    faiss_mapping_path: str | Path,
    lora_adapter_dir: str | Path,
    distilbert_dir: str | Path | None = None,
    xgb_model=None,
    embedder: SentenceTransformer | None = None,
    n_samples: int = 100,
    seed: int = 42,
    rag_top_k: int = 3,
    threshold_xgb: float = 0.70,
    threshold_bert: float = 0.75,
    threshold_lora: float = 0.75,
    generator_model_id: str | None = None,
) -> dict:
    """
    If distilbert_dir is None, skips D_BERT. If xgb_model/embedder None, skips C_XGB.
    xgb_model must be fitted XGBClassifier from router_baselines (expects embedder in scope).
    If generator_model_id is None, uses ``DEFAULT_GENERATOR_MODEL`` (must match the LoRA base).
    """
    model_id = generator_model_id or DEFAULT_GENERATOR_MODEL
    embedder_id = DEFAULT_EMBEDDER_MODEL

    tokenizer, gen_model = load_generator(model_id)
    rag_index = faiss.read_index(str(faiss_index_path))
    with open(faiss_mapping_path, "rb") as f:
        rag_mapping = pickle.load(f)

    if embedder is None:
        embedder = SentenceTransformer(
            embedder_id, device="cuda:0" if torch.cuda.is_available() else "cpu"
        )

    lora_tok, lora_router = load_lora_router(model_id, lora_adapter_dir)
    gen_device = next(gen_model.parameters()).device

    bert_tok = bert_router = None
    if distilbert_dir and Path(distilbert_dir).exists():
        bert_tok, bert_router = load_distilbert_router(distilbert_dir)

    def system_a(question: str):
        t0 = time.time()
        prompt = f"Question: {question}\n\nProvide a concise medical answer:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_device)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        ans = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        return ans, time.time() - t0, n_in

    def system_b(question: str):
        t0 = time.time()
        assert embedder is not None
        qe = np.array([embedder.encode(question)]).astype("float32")
        faiss.normalize_L2(qe)
        _, idx = rag_index.search(qe, rag_top_k)
        retrieved = "\n\n".join(rag_mapping["texts"][int(i)] for i in idx[0])
        prompt = (
            f"Context: {retrieved}\n\nQuestion: {question}\n\n"
            "Based strictly on the context, provide a concise medical answer:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_device)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        ans = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        return ans, time.time() - t0, n_in

    def route_xgb(question: str):
        assert xgb_model is not None and embedder is not None
        t0 = time.time()
        qe = embedder.encode([question], convert_to_numpy=True)
        p1 = float(xgb_model.predict_proba(qe)[0][1])
        pred = 1 if p1 > threshold_xgb else 0
        return pred, time.time() - t0

    def route_bert(question: str):
        assert bert_tok is not None and bert_router is not None
        t0 = time.time()
        dev = next(bert_router.parameters()).device
        inputs = bert_tok(question, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            logits = bert_router(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred = 1 if probs[0][1].item() > threshold_bert else 0
        return pred, time.time() - t0

    def route_lora(question: str):
        t0 = time.time()
        prompt = f"Context: \n\nQuestion: {question}\n\nProvide a concise medical answer:\n"
        ldev = next(lora_router.parameters()).device
        inputs = lora_tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(ldev)
        with torch.no_grad():
            logits = lora_router(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred = 1 if probs[0][1].item() > threshold_lora else 0
        return pred, time.time() - t0

    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    df_eval = pd.DataFrame(dataset)
    df_test = df_eval.sample(n=min(n_samples, len(df_eval)), random_state=seed).reset_index(drop=True)

    metrics = {
        "A": {"correct": 0, "time": 0.0, "tokens": 0},
        "B": {"correct": 0, "time": 0.0, "tokens": 0},
    }
    if xgb_model is not None:
        metrics["C_XGB"] = {"correct": 0, "time": 0.0, "tokens": 0, "bypassed": 0}
    if bert_router is not None:
        metrics["D_BERT"] = {"correct": 0, "time": 0.0, "tokens": 0, "bypassed": 0}
    metrics["E_LORA"] = {"correct": 0, "time": 0.0, "tokens": 0, "bypassed": 0}

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        q = row["question"]
        truth = str(row["final_decision"]).strip().lower()

        ans_a, t_a, tok_a = system_a(q)
        ans_b, t_b, tok_b = system_b(q)
        ok_a = truth in ans_a.lower()
        ok_b = truth in ans_b.lower()

        metrics["A"]["time"] += t_a
        metrics["A"]["tokens"] += tok_a
        if ok_a:
            metrics["A"]["correct"] += 1
        metrics["B"]["time"] += t_b
        metrics["B"]["tokens"] += tok_b
        if ok_b:
            metrics["B"]["correct"] += 1

        if "C_XGB" in metrics:
            px, tx = route_xgb(q)
            metrics["C_XGB"]["time"] += tx
            if px == 0:
                metrics["C_XGB"]["time"] += t_a
                metrics["C_XGB"]["tokens"] += tok_a
                metrics["C_XGB"]["bypassed"] += 1
                if ok_a:
                    metrics["C_XGB"]["correct"] += 1
            else:
                metrics["C_XGB"]["time"] += t_b
                metrics["C_XGB"]["tokens"] += tok_b
                if ok_b:
                    metrics["C_XGB"]["correct"] += 1

        if "D_BERT" in metrics:
            pb, tb = route_bert(q)
            metrics["D_BERT"]["time"] += tb
            if pb == 0:
                metrics["D_BERT"]["time"] += t_a
                metrics["D_BERT"]["tokens"] += tok_a
                metrics["D_BERT"]["bypassed"] += 1
                if ok_a:
                    metrics["D_BERT"]["correct"] += 1
            else:
                metrics["D_BERT"]["time"] += t_b
                metrics["D_BERT"]["tokens"] += tok_b
                if ok_b:
                    metrics["D_BERT"]["correct"] += 1

        pl, tl = route_lora(q)
        metrics["E_LORA"]["time"] += tl
        if pl == 0:
            metrics["E_LORA"]["time"] += t_a
            metrics["E_LORA"]["tokens"] += tok_a
            metrics["E_LORA"]["bypassed"] += 1
            if ok_a:
                metrics["E_LORA"]["correct"] += 1
        else:
            metrics["E_LORA"]["time"] += t_b
            metrics["E_LORA"]["tokens"] += tok_b
            if ok_b:
                metrics["E_LORA"]["correct"] += 1

    total = len(df_test)
    summary = {"n": total, "metrics": metrics}
    for k, m in metrics.items():
        m["accuracy"] = m["correct"] / total if total else 0.0
        if "bypassed" in m:
            m["bypass_frac"] = m["bypassed"] / total if total else 0.0
    return summary


def print_summary(summary: dict) -> None:
    n = summary["n"]
    m = summary["metrics"]
    print("\n" + "=" * 60)
    print("Adaptive RAG shootout (PubMedQA yes/no in answer string)")
    print("=" * 60)
    for name, key in [
        ("Vanilla LLM", "A"),
        ("Always RAG", "B"),
        ("XGBoost router", "C_XGB"),
        ("DistilBERT router", "D_BERT"),
        ("Llama LoRA router", "E_LORA"),
    ]:
        if key not in m:
            continue
        row = m[key]
        extra = ""
        if "bypass_frac" in row:
            extra = f" | RAG bypass rate: {row['bypass_frac']*100:.1f}%"
        print(f"{name:22} acc={row['accuracy']*100:.1f}%{extra} | latency_s={row['time']:.2f} | tokens={row['tokens']}")
    print("=" * 60)
