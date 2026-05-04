"""Ensure pipeline artifacts exist (MedHallu CSV, LoRA, FAISS, XGB) by running scripts 02–05 when needed."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from adaptive_rag.config import (
    ARTIFACTS_DIR,
    LORA_ROUTER_DIR,
    MEDHALLU_TRAINING_CSV,
    PUBMED_FAISS_INDEX,
    PUBMED_FAISS_MAPPING,
    REPO_ROOT,
)


def xgb_router_path() -> Path:
    return ARTIFACTS_DIR / "xgb_router.joblib"


def lora_ready() -> bool:
    return LORA_ROUTER_DIR.is_dir() and (LORA_ROUTER_DIR / "adapter_config.json").is_file()


def faiss_ready() -> bool:
    return PUBMED_FAISS_INDEX.is_file() and PUBMED_FAISS_MAPPING.is_file()


def list_missing_final_artifacts() -> list[str]:
    missing: list[str] = []
    if not MEDHALLU_TRAINING_CSV.is_file():
        missing.append(f"MedHallu CSV ({MEDHALLU_TRAINING_CSV})")
    if not lora_ready():
        missing.append(f"LoRA adapter ({LORA_ROUTER_DIR} + adapter_config.json)")
    if not faiss_ready():
        missing.append(f"FAISS index/mapping ({PUBMED_FAISS_INDEX}, {PUBMED_FAISS_MAPPING})")
    if not xgb_router_path().is_file():
        missing.append(f"XGBoost router ({xgb_router_path()})")
    return missing


def _run_script(repo_root: Path, script: str, args: Sequence[str], *, hf_token: str | None) -> None:
    cmd = [sys.executable, str(repo_root / "scripts" / script), *args]
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Script failed with exit code {proc.returncode}: {' '.join(cmd)}")


def ensure_final_artifacts(
    repo_root: Path | None = None,
    *,
    skip_build: bool = False,
    hf_token: str | None = None,
    medhallu_batch_size: int = 16,
    medhallu_model_id: str | None = None,
    lora_epochs: int = 2,
    faiss_batch_size: int = 256,
    skip_distilbert: bool = False,
) -> None:
    """
    Ensure CSV, LoRA, FAISS, and XGBoost joblib exist; run scripts 02→05 in order when something is missing.

    Uses paths from ``adaptive_rag.config`` (respect ``ADAPTIVE_RAG_DATA`` / ``ADAPTIVE_RAG_ARTIFACTS`` if set).
    ``repo_root`` is only used to locate ``scripts/``; must match the repo that owns those paths.
    """
    root = repo_root or REPO_ROOT
    missing = list_missing_final_artifacts()
    if not missing:
        return
    if skip_build:
        raise FileNotFoundError(
            "Missing required artifacts (--skip-build). Missing:\n  - " + "\n  - ".join(missing)
        )

    extra_02: list[str] = ["--batch-size", str(medhallu_batch_size)]
    if medhallu_model_id:
        extra_02 += ["--model-id", medhallu_model_id]
    if not MEDHALLU_TRAINING_CSV.is_file():
        print("--- (1/4) MedHallu CSV ---")
        _run_script(root, "02_generate_medhallu_router_csv.py", extra_02, hf_token=hf_token)

    if not lora_ready():
        print("--- (2/4) LoRA router ---")
        _run_script(
            root,
            "03_train_lora_router.py",
            ["--epochs", str(lora_epochs)],
            hf_token=hf_token,
        )

    if not faiss_ready():
        print("--- (3/4) PubMed FAISS ---")
        _run_script(
            root,
            "04_build_pubmed_faiss.py",
            ["--batch-size", str(faiss_batch_size)],
            hf_token=hf_token,
        )

    if not xgb_router_path().is_file():
        print("--- (4/4) Baseline routers (XGB + optional DistilBERT) ---")
        args05 = []
        if skip_distilbert:
            args05.append("--skip-bert")
        _run_script(root, "05_train_baseline_routers.py", args05, hf_token=hf_token)

    still = list_missing_final_artifacts()
    if still:
        raise RuntimeError("Artifacts still missing after build:\n  - " + "\n  - ".join(still))
