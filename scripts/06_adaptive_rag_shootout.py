#!/usr/bin/env python3
"""Step 6: PubMedQA benchmark — vanilla vs always-RAG vs learned routers (LoRA + optional XGB/DistilBERT)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import torch
from huggingface_hub import login

from adaptive_rag.config import (
    ARTIFACTS_DIR,
    DISTILBERT_ROUTER_DIR,
    LORA_ROUTER_DIR,
    PUBMED_FAISS_INDEX,
    PUBMED_FAISS_MAPPING,
)
from adaptive_rag.report_benchmark import run_shootout_report_cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss-index", type=Path, default=PUBMED_FAISS_INDEX)
    ap.add_argument("--faiss-mapping", type=Path, default=PUBMED_FAISS_MAPPING)
    ap.add_argument("--lora-adapter", type=Path, default=LORA_ROUTER_DIR)
    ap.add_argument("--distilbert-dir", type=Path, default=DISTILBERT_ROUTER_DIR)
    ap.add_argument("--xgb-path", type=Path, default=ARTIFACTS_DIR / "xgb_router.joblib")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--no-xgb", action="store_true")
    ap.add_argument("--no-bert", action="store_true")
    ap.add_argument("--csv-out", type=Path, default=None, help="Optional path to save shootout table (CSV)")
    ap.add_argument(
        "--generator-model",
        default=None,
        help="Causal LM + LoRA base (default: ADAPTIVE_RAG_GENERATOR_MODEL / config; must match adapter training).",
    )
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    if not args.faiss_index.exists() or not args.faiss_mapping.exists():
        raise SystemExit("FAISS index/mapping missing. Run scripts/04_build_pubmed_faiss.py")
    if not args.lora_adapter.exists():
        raise SystemExit("LoRA adapter missing. Run scripts/03_train_lora_router.py (or download weights).")

    xgb_model = None
    if not args.no_xgb and args.xgb_path.exists():
        xgb_model = joblib.load(args.xgb_path)
    elif not args.no_xgb:
        print("Note: XGBoost artifact not found; run scripts/05_train_baseline_routers.py or pass --no-xgb")

    distil_dir = None if args.no_bert or not args.distilbert_dir.exists() else args.distilbert_dir
    if not args.no_bert and distil_dir is None:
        print("Note: DistilBERT router dir missing; skipping D_BERT (train with step 5 or pass --no-bert)")

    kw: dict = dict(
        faiss_index_path=args.faiss_index,
        faiss_mapping_path=args.faiss_mapping,
        lora_adapter_dir=args.lora_adapter,
        distilbert_dir=distil_dir,
        xgb_model=xgb_model,
        n_samples=args.n_samples,
        print_table=True,
        csv_path=args.csv_out,
    )
    if args.generator_model:
        kw["generator_model_id"] = args.generator_model
    run_shootout_report_cli(**kw)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Large generator models will be slow or may OOM on CPU.")
    main()
