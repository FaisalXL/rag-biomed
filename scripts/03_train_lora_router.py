#!/usr/bin/env python3
"""Step 3: PEFT LoRA sequence classifier on Llama 3.1 8B using CSV from step 2."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import login

from adaptive_rag.config import LORA_ROUTER_DIR, MEDHALLU_TRAINING_CSV
from adaptive_rag.train_lora_router import train_lora_router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=MEDHALLU_TRAINING_CSV)
    ap.add_argument("--output-dir", type=Path, default=LORA_ROUTER_DIR)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument(
        "--model-id",
        default=None,
        help="Base causal LM for seq-cls + LoRA (default: ADAPTIVE_RAG_GENERATOR_MODEL / config).",
    )
    ap.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        metavar="N",
        help="If set, stop after N optimizer steps (dev check; disables per-epoch eval/save-best).",
    )
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    if not args.csv.exists():
        raise SystemExit(f"Missing training CSV: {args.csv}. Run scripts/02_generate_medhallu_router_csv.py first.")

    train_lora_router(
        csv_path=args.csv,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        model_id=args.model_id,
        max_train_steps=args.max_train_steps,
    )


if __name__ == "__main__":
    main()
