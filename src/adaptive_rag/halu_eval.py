"""Load HaluEval (pminervini/HaluEval) and normalize columns for binary classification."""

from __future__ import annotations

import pandas as pd
from datasets import DatasetDict, load_dataset


def _halu_eval_frame(task_name: str) -> pd.DataFrame:
    raw = load_dataset("pminervini/HaluEval", task_name)
    if not isinstance(raw, DatasetDict):
        raise TypeError("Expected DatasetDict from HaluEval")
    if "train" in raw:
        return pd.DataFrame(raw["train"])
    if "data" in raw:
        return pd.DataFrame(raw["data"])
    first = next(iter(raw.keys()))
    return pd.DataFrame(raw[first])


def load_and_standardize_halu_eval(task_name: str) -> pd.DataFrame:
    """
    Returns shuffled dataframe with columns: context, response, label
    (0 = factual, 1 = hallucinated).
    """
    df = _halu_eval_frame(task_name)

    if task_name == "qa":
        col_context, col_right, col_hallu = (
            "question",
            "right_answer",
            "hallucinated_answer",
        )
    elif task_name == "dialogue":
        col_context, col_right, col_hallu = (
            "dialogue_history",
            "right_response",
            "hallucinated_response",
        )
    elif task_name == "summarization":
        col_context, col_right, col_hallu = (
            "document",
            "right_summary",
            "hallucinated_summary",
        )
    else:
        raise ValueError(f"Unknown HaluEval task: {task_name}")

    df_factual = df[[col_context, col_right]].copy()
    df_factual.columns = ["context", "response"]
    df_factual["label"] = 0

    df_hallu = df[[col_context, col_hallu]].copy()
    df_hallu.columns = ["context", "response"]
    df_hallu["label"] = 1

    return (
        pd.concat([df_factual, df_hallu])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def format_context_response_row(context: str, response: str) -> str:
    return f"Context: {context}\nOutput: {response}"
