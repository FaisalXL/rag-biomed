#!/usr/bin/env python3
"""Step 2: build `data/medhallu_lora_training_data_batch.csv` (prompt, label) from MedHallu."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import login

from adaptive_rag.config import DEFAULT_GENERATOR_MODEL, MEDHALLU_TRAINING_CSV
from adaptive_rag.medhallu_data_gen import build_medhallu_router_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=MEDHALLU_TRAINING_CSV)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model-id", default=os.environ.get("ADAPTIVE_RAG_GENERATOR_MODEL", DEFAULT_GENERATOR_MODEL))
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="If set, only the first N MedHallu rows are used (default: full split).",
    )
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    build_medhallu_router_csv(
        output_path=args.output,
        batch_size=args.batch_size,
        model_id=args.model_id,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
