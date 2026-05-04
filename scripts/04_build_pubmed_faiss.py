#!/usr/bin/env python3
"""Step 4: embed PubMedQA abstracts (all splits, deduped) and write FAISS index + id/text mapping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config import PUBMED_FAISS_INDEX, PUBMED_FAISS_MAPPING
from adaptive_rag.pubmed_faiss import build_pubmed_faiss_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=PUBMED_FAISS_INDEX)
    ap.add_argument("--mapping", type=Path, default=PUBMED_FAISS_MAPPING)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    build_pubmed_faiss_index(
        index_path=args.index,
        mapping_path=args.mapping,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
