#!/usr/bin/env python3
"""Optional: classical HaluEval baselines (matches legacy Brouge / ngram_basic notebooks)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.text_baselines import run_ngram_mlp, run_tfidf_svm_xgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="qa", choices=["qa", "dialogue", "summarization"])
    ap.add_argument("--method", choices=["tfidf", "ngram"], default="tfidf")
    args = ap.parse_args()

    if args.method == "tfidf":
        run_tfidf_svm_xgb(args.task)
    else:
        run_ngram_mlp(args.task)


if __name__ == "__main__":
    main()
