"""Linear probe on last-token hidden states (HaluEval) — motivation for router signal."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_rag.halu_eval import format_context_response_row, load_and_standardize_halu_eval


def run_hidden_state_probe(
    model_id: str,
    tasks: list[str] | None = None,
    max_samples: int | None = None,
    max_length: int = 2048,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    tasks = tasks or ["qa", "dialogue", "summarization"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()

    for task in tasks:
        df_task = load_and_standardize_halu_eval(task)
        if max_samples is not None:
            df_task = df_task.head(max_samples).reset_index(drop=True)

        X_list: list[np.ndarray] = []
        y = df_task["label"].tolist()

        for _, row in tqdm(
            df_task.iterrows(),
            total=len(df_task),
            desc=f"hidden states [{task}]",
        ):
            text = format_context_response_row(str(row["context"]), str(row["response"]))
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)

            last = outputs.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
            X_list.append(last)

        X = np.asarray(X_list)
        y_arr = np.asarray(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_arr, test_size=test_size, random_state=random_state
        )

        clf = LogisticRegression(max_iter=2000, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        print(f"\n--- RESULTS: {task.upper()} | model={model_id} ---")
        print(classification_report(y_test, y_pred, target_names=["Factual (0)", "Hallucination (1)"]))
        if len(np.unique(y_test)) > 1:
            print(f"AUROC: {roc_auc_score(y_test, y_prob):.4f}")
            print(f"AUPRC: {average_precision_score(y_test, y_prob):.4f}")
