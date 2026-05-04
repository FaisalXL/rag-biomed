"""Build tabular reports from `run_shootout` summaries for notebooks, CLI, and CSV export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_rag.adaptive_shootout import print_summary, run_shootout


def shootout_summary_to_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Turn a `run_shootout` return value into one row per system."""
    n = summary["n"]
    metrics = summary["metrics"]
    rows: list[dict[str, Any]] = []
    labels = {
        "A": "Vanilla_LLM",
        "B": "Always_RAG",
        "C_XGB": "Adaptive_XGBoost",
        "D_BERT": "Adaptive_DistilBERT",
        "E_LORA": "Adaptive_LoRA",
    }
    for key, name in labels.items():
        if key not in metrics:
            continue
        m = metrics[key]
        row: dict[str, Any] = {
            "system": name,
            "n_questions": n,
            "accuracy": m["accuracy"],
            "latency_s": m["time"],
            "prompt_tokens": m["tokens"],
        }
        if "bypass_frac" in m:
            row["rag_bypass_rate"] = m["bypass_frac"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_shootout_report(
    *,
    faiss_index_path: str | Path,
    faiss_mapping_path: str | Path,
    lora_adapter_dir: str | Path,
    distilbert_dir: str | Path | None = None,
    xgb_model=None,
    n_samples: int = 100,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Run PubMedQA adaptive shootout and return (raw_summary, dataframe).
    Pass-through kwargs go to `run_shootout` (e.g. threshold_xgb, rag_top_k).
    """
    summary = run_shootout(
        faiss_index_path=faiss_index_path,
        faiss_mapping_path=faiss_mapping_path,
        lora_adapter_dir=lora_adapter_dir,
        distilbert_dir=distilbert_dir,
        xgb_model=xgb_model,
        n_samples=n_samples,
        seed=seed,
        **kwargs,
    )
    return summary, shootout_summary_to_dataframe(summary)


def run_shootout_report_cli(
    *,
    print_table: bool = True,
    csv_path: str | Path | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Convenience: run report, optionally print `print_summary` and save CSV."""
    summary, df = run_shootout_report(**kwargs)
    if print_table:
        print_summary(summary)
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
        print(f"Wrote table to {csv_path}")
    return summary, df
