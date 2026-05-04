#!/usr/bin/env python3
"""One optimizer step through train_lora_router (tiny CSV, temp output). Checks PEFT/Trainer/MPS/CUDA path."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from huggingface_hub import login

    from adaptive_rag.config import DEFAULT_GENERATOR_MODEL
    from adaptive_rag.train_lora_router import train_lora_router

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id",
        default=os.environ.get("ADAPTIVE_RAG_GENERATOR_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    )
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--max-steps", type=int, default=1, help="Optimizer steps (default 1).")
    args = ap.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    rows = 32
    df = pd.DataFrame(
        {
            "prompt": [f"Question {i}: test?\n\nProvide a concise medical answer:\n" for i in range(rows)],
            "label": [i % 2 for i in range(rows)],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
    df.to_csv(csv_path, index=False)

    out = Path(tempfile.mkdtemp(prefix="verify_lora_"))
    try:
        train_lora_router(
            csv_path=csv_path,
            output_dir=out,
            model_id=args.model_id or DEFAULT_GENERATOR_MODEL,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            max_train_steps=args.max_steps,
        )
    finally:
        csv_path.unlink(missing_ok=True)

    print("verify_minimal_train: OK (weights under", out, ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
