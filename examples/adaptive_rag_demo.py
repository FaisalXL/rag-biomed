#!/usr/bin/env python3
"""Shorter adaptive shootout demo. Paired with `notebooks/adaptive_rag_demo.ipynb`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import torch
    from huggingface_hub import login

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=25, help="PubMedQA subsample size")
    ap.add_argument("--csv-out", type=Path, default=None, help="Optional CSV for results table")
    ap.add_argument(
        "--no-train-xgb",
        action="store_true",
        help="Fail if xgb_router.joblib missing instead of training",
    )
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    os.chdir(ROOT)
    if args.hf_token:
        login(token=args.hf_token)

    if not torch.cuda.is_available():
        print("Warning: CUDA not available; Llama 8B may be impractical.")

    from adaptive_rag.demo_adaptive import run_adaptive_shootout_demo

    run_adaptive_shootout_demo(
        n_samples=args.n_samples,
        train_xgb_if_missing=not args.no_train_xgb,
        csv_path=args.csv_out,
        print_table=True,
    )


if __name__ == "__main__":
    main()
