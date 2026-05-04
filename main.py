#!/usr/bin/env python3
"""Repo entrypoint: final PubMedQA shootout (with optional artifact build), probe, or text baselines."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def cmd_final(args: argparse.Namespace) -> int:
    import joblib
    import torch
    from huggingface_hub import login

    from adaptive_rag.config import (
        ARTIFACTS_DIR,
        DEFAULT_GENERATOR_MODEL,
        DISTILBERT_ROUTER_DIR,
        LORA_ROUTER_DIR,
        PUBMED_FAISS_INDEX,
        PUBMED_FAISS_MAPPING,
    )
    from adaptive_rag.orchestrate import ensure_final_artifacts
    from adaptive_rag.report_benchmark import run_shootout_report_cli

    hf = args.hf_token or os.environ.get("HF_TOKEN")
    if hf:
        login(token=hf)

    medhallu_model = args.model_id or os.environ.get("ADAPTIVE_RAG_GENERATOR_MODEL", DEFAULT_GENERATOR_MODEL)

    ensure_final_artifacts(
        ROOT,
        skip_build=args.skip_build,
        hf_token=hf,
        medhallu_batch_size=args.batch_size,
        medhallu_model_id=medhallu_model,
        lora_epochs=args.epochs,
        faiss_batch_size=args.faiss_batch_size,
        skip_distilbert=args.skip_bert,
    )

    if not args.faiss_index.exists() or not args.faiss_mapping.exists():
        print("FAISS index/mapping missing.", file=sys.stderr)
        return 1
    if not args.lora_adapter.exists():
        print("LoRA adapter missing.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Llama 8B will be slow or OOM on CPU.")

    xgb_model = None
    if not args.no_xgb and args.xgb_path.exists():
        xgb_model = joblib.load(args.xgb_path)
    elif not args.no_xgb:
        print("Note: XGBoost artifact not found; run scripts/05 or pass --no-xgb")

    distil_dir = None if args.no_bert or not args.distilbert_dir.exists() else args.distilbert_dir
    if not args.no_bert and distil_dir is None:
        print("Note: DistilBERT router dir missing; skipping D_BERT (train with step 5 or pass --no-bert)")

    run_shootout_report_cli(
        faiss_index_path=args.faiss_index,
        faiss_mapping_path=args.faiss_mapping,
        lora_adapter_dir=args.lora_adapter,
        distilbert_dir=distil_dir,
        xgb_model=xgb_model,
        n_samples=args.n_samples,
        print_table=True,
        csv_path=args.csv_out,
    )
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from huggingface_hub import login

    hf = args.hf_token or os.environ.get("HF_TOKEN")
    if hf:
        login(token=hf)
    cmd: list[str] = [
        sys.executable,
        str(ROOT / "scripts" / "01_internal_state_probe.py"),
        "--model-id",
        args.model_id,
    ]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    cmd += ["--tasks", *args.tasks]
    if hf:
        cmd.extend(["--hf-token", hf])
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_text_baselines(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "00_text_baselines_haluval.py"),
        "--task",
        args.task,
        "--method",
        args.method,
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adaptive RAG pipeline entrypoint.")
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("final", help="Build missing artifacts (02–05) then run PubMedQA shootout.")
    pf.add_argument("--skip-build", action="store_true", help="Fail if CSV/LoRA/FAISS/XGB are missing.")
    pf.add_argument("--hf-token", default=None, help="Hugging Face token (else HF_TOKEN env).")
    pf.add_argument("--batch-size", type=int, default=16, help="MedHallu generation batch size (script 02).")
    pf.add_argument("--faiss-batch-size", type=int, default=256, help="FAISS embedding batch size (script 04).")
    pf.add_argument("--epochs", type=int, default=2, help="LoRA training epochs (script 03).")
    pf.add_argument(
        "--model-id",
        default=None,
        help="Generator for MedHallu CSV (default: config / ADAPTIVE_RAG_GENERATOR_MODEL).",
    )
    pf.add_argument("--skip-bert", action="store_true", help="Pass --skip-bert to script 05.")
    pf.add_argument("--n-samples", type=int, default=100)
    pf.add_argument("--csv-out", type=Path, default=None)
    pf.add_argument("--no-xgb", action="store_true")
    pf.add_argument("--no-bert", action="store_true")
    pf.add_argument("--faiss-index", type=Path, default=None)
    pf.add_argument("--faiss-mapping", type=Path, default=None)
    pf.add_argument("--lora-adapter", type=Path, default=None)
    pf.add_argument("--distilbert-dir", type=Path, default=None)
    pf.add_argument("--xgb-path", type=Path, default=None)
    pf.set_defaults(_handler=cmd_final)

    pp = sub.add_parser("probe", help="Run hidden-state linear probe (HaluEval).")
    pp.add_argument(
        "--model-id",
        default=os.environ.get("PROBE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"),
    )
    pp.add_argument("--tasks", nargs="*", default=["qa", "dialogue", "summarization"])
    pp.add_argument("--max-samples", type=int, default=None)
    pp.add_argument("--hf-token", default=None)
    pp.set_defaults(_handler=cmd_probe)

    pt = sub.add_parser("text-baselines", help="HaluEval TF-IDF or n-gram baselines.")
    pt.add_argument("--task", default="qa", choices=["qa", "dialogue", "summarization"])
    pt.add_argument("--method", choices=["tfidf", "ngram"], default="tfidf")
    pt.set_defaults(_handler=cmd_text_baselines)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    from adaptive_rag.config import (
        ARTIFACTS_DIR,
        DISTILBERT_ROUTER_DIR,
        LORA_ROUTER_DIR,
        PUBMED_FAISS_INDEX,
        PUBMED_FAISS_MAPPING,
    )

    if args.command == "final":
        if args.faiss_index is None:
            args.faiss_index = PUBMED_FAISS_INDEX
        if args.faiss_mapping is None:
            args.faiss_mapping = PUBMED_FAISS_MAPPING
        if args.lora_adapter is None:
            args.lora_adapter = LORA_ROUTER_DIR
        if args.distilbert_dir is None:
            args.distilbert_dir = DISTILBERT_ROUTER_DIR
        if args.xgb_path is None:
            args.xgb_path = ARTIFACTS_DIR / "xgb_router.joblib"

    return int(args._handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
