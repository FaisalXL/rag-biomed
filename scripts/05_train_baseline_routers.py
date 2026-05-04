#!/usr/bin/env python3
"""Step 5 (optional): train XGBoost + DistilBERT routers on the same CSV; save XGBoost to artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from huggingface_hub import login

from adaptive_rag.config import ARTIFACTS_DIR, DISTILBERT_ROUTER_DIR, MEDHALLU_TRAINING_CSV
from adaptive_rag.router_baselines import train_distilbert_router, train_xgboost_router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=MEDHALLU_TRAINING_CSV)
    ap.add_argument("--xgb-out", type=Path, default=ARTIFACTS_DIR / "xgb_router.joblib")
    ap.add_argument("--bert-out", type=Path, default=DISTILBERT_ROUTER_DIR)
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    try:
        df = pd.read_csv(args.csv, lineterminator="\n")
    except Exception:
        df = pd.read_csv(args.csv, engine="python", on_bad_lines="skip")

    prompts = df["prompt"].tolist()
    labels = df["label"].tolist()

    clf, _emb = train_xgboost_router(prompts, labels)
    args.xgb_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.xgb_out)
    print(f"Saved XGBoost router to {args.xgb_out}")

    if not args.skip_bert:
        train_distilbert_router(prompts, labels, output_dir=args.bert_out)


if __name__ == "__main__":
    main()
