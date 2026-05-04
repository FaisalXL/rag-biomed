"""Paths and defaults. Override with environment variables for your machine or Colab."""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of src/
REPO_ROOT = Path(__file__).resolve().parents[2]

# Artifacts (FAISS, CSVs, adapters)
ARTIFACTS_DIR = Path(os.environ.get("ADAPTIVE_RAG_ARTIFACTS", REPO_ROOT / "artifacts"))
DATA_DIR = Path(os.environ.get("ADAPTIVE_RAG_DATA", REPO_ROOT / "data"))

# Models (gated Llama needs HF_TOKEN)
DEFAULT_GENERATOR_MODEL = os.environ.get(
    "ADAPTIVE_RAG_GENERATOR_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_EMBEDDER_MODEL = os.environ.get(
    "ADAPTIVE_RAG_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"
)

MEDHALLU_TRAINING_CSV = DATA_DIR / "medhallu_lora_training_data_batch.csv"
PUBMED_FAISS_INDEX = ARTIFACTS_DIR / "pubmed_massive_faiss.index"
PUBMED_FAISS_MAPPING = ARTIFACTS_DIR / "pubmed_massive_mapping.pkl"
LORA_ROUTER_DIR = ARTIFACTS_DIR / "llama3-medhallu-router"
DISTILBERT_ROUTER_DIR = ARTIFACTS_DIR / "distilbert_router"


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
