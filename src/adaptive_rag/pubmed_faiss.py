"""Build FAISS index over PubMedQA abstracts (all subsets, deduped by pubid)."""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from adaptive_rag.config import DEFAULT_EMBEDDER_MODEL, PUBMED_FAISS_INDEX, PUBMED_FAISS_MAPPING, ensure_dirs


def build_pubmed_faiss_index(
    index_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    embedder_id: str | None = None,
    batch_size: int = 256,
) -> tuple[Path, Path]:
    ensure_dirs()
    index_path = Path(index_path or PUBMED_FAISS_INDEX)
    mapping_path = Path(mapping_path or PUBMED_FAISS_MAPPING)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    embedder_id = embedder_id or DEFAULT_EMBEDDER_MODEL

    ds_labeled = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    ds_artificial = load_dataset("pubmed_qa", "pqa_artificial", split="train")
    ds_unlabeled = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")

    doc_lib: dict[str, str] = {}
    for ds in [ds_labeled, ds_artificial, ds_unlabeled]:
        for row in tqdm(ds, desc="collect abstracts"):
            full_abs = " ".join(row["context"]["contexts"])
            pid = str(row["pubid"])
            if pid not in doc_lib:
                doc_lib[pid] = full_abs

    doc_ids = list(doc_lib.keys())
    doc_texts = list(doc_lib.values())

    embedder = SentenceTransformer(embedder_id)
    dim = embedder.get_sentence_embedding_dimension()
    embs = embedder.encode(doc_texts, show_progress_bar=True, batch_size=batch_size)
    embs = np.asarray(embs, dtype=np.float32)
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, str(index_path))
    with open(mapping_path, "wb") as f:
        pickle.dump({"ids": doc_ids, "texts": doc_texts}, f)

    print(f"Wrote index ({index.ntotal} vectors) -> {index_path}")
    print(f"Wrote mapping -> {mapping_path}")
    return index_path, mapping_path
