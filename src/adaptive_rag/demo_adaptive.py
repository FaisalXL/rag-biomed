"""Small runnable demo for portfolio notebooks and `examples/adaptive_rag_demo.py` (shorter shootout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from adaptive_rag.config import (
    ARTIFACTS_DIR,
    DISTILBERT_ROUTER_DIR,
    LORA_ROUTER_DIR,
    MEDHALLU_TRAINING_CSV,
    PUBMED_FAISS_INDEX,
    PUBMED_FAISS_MAPPING,
)
from adaptive_rag.report_benchmark import run_shootout_report_cli


def run_adaptive_shootout_demo(
    n_samples: int = 25,
    *,
    train_xgb_if_missing: bool = True,
    csv_path: Path | None = None,
    print_table: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Load or train XGBoost router, then run adaptive shootout (requires FAISS + LoRA on disk).
    Uses a smaller default `n_samples` than the full paper run for quicker portfolio demos.
    """
    if not PUBMED_FAISS_INDEX.exists() or not PUBMED_FAISS_MAPPING.exists():
        raise FileNotFoundError("Build FAISS first: python scripts/04_build_pubmed_faiss.py")
    if not LORA_ROUTER_DIR.exists():
        raise FileNotFoundError("Train LoRA first: python scripts/03_train_lora_router.py")
    if not MEDHALLU_TRAINING_CSV.exists():
        raise FileNotFoundError("Create CSV first: python scripts/02_generate_medhallu_router_csv.py")

    xgb_path = ARTIFACTS_DIR / "xgb_router.joblib"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    xgb_model = None
    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
    elif train_xgb_if_missing:
        from adaptive_rag.router_baselines import train_xgboost_router

        df = pd.read_csv(MEDHALLU_TRAINING_CSV, engine="python", on_bad_lines="skip")
        xgb_model, _ = train_xgboost_router(df["prompt"].tolist(), df["label"].tolist())
        joblib.dump(xgb_model, xgb_path)

    distil_dir = DISTILBERT_ROUTER_DIR if DISTILBERT_ROUTER_DIR.exists() else None

    return run_shootout_report_cli(
        faiss_index_path=PUBMED_FAISS_INDEX,
        faiss_mapping_path=PUBMED_FAISS_MAPPING,
        lora_adapter_dir=LORA_ROUTER_DIR,
        distilbert_dir=distil_dir,
        xgb_model=xgb_model,
        n_samples=n_samples,
        print_table=print_table,
        csv_path=csv_path,
    )
