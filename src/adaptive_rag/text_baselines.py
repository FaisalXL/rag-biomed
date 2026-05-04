"""Classical text baselines on HaluEval (TF-IDF + SVM/XGBoost, N-gram + SVD + MLP). Kept for report continuity."""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from adaptive_rag.halu_eval import _halu_eval_frame


def halu_eval_text_pairs(task_name: str, max_rows: int | None = 10_000):
    df = _halu_eval_frame(task_name)
    if task_name == "qa":
        factual = df[["question", "right_answer"]].copy()
        factual.columns = ["prompt", "response"]
        factual["label"] = 0
        hallu = df[["question", "hallucinated_answer"]].copy()
        hallu.columns = ["prompt", "response"]
        hallu["label"] = 1
    elif task_name == "dialogue":
        factual = df[["dialogue_history", "right_response"]].copy()
        factual.columns = ["prompt", "response"]
        factual["label"] = 0
        hallu = df[["dialogue_history", "hallucinated_response"]].copy()
        hallu.columns = ["prompt", "response"]
        hallu["label"] = 1
    elif task_name == "summarization":
        factual = df[["document", "right_summary"]].copy()
        factual.columns = ["prompt", "response"]
        factual["label"] = 0
        hallu = df[["document", "hallucinated_summary"]].copy()
        hallu.columns = ["prompt", "response"]
        hallu["label"] = 1
    else:
        raise ValueError(task_name)

    comb = pd.concat([factual, hallu]).sample(frac=1, random_state=42).reset_index(drop=True)
    comb["text_to_analyze"] = comb["prompt"] + " [SEP] " + comb["response"]
    if max_rows is not None:
        comb = comb.head(max_rows)
    return comb


def run_tfidf_svm_xgb(task_name: str = "qa") -> None:
    comb = halu_eval_text_pairs(task_name)
    X_train, X_test, y_train, y_test = train_test_split(
        comb["text_to_analyze"], comb["label"], test_size=0.2, random_state=42
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), max_df=0.90, min_df=5, max_features=10000)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    svm = SVC(kernel="linear", probability=True, random_state=42)
    svm.fit(Xtr, y_train)
    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    xgb.fit(Xtr, y_train)

    for name, model in [("SVM", svm), ("XGBoost", xgb)]:
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        print(f"\n--- {name} | HaluEval {task_name} ---")
        print(classification_report(y_test, y_pred, target_names=["Factual (0)", "Hallucination (1)"]))
        print(f"AUROC: {roc_auc_score(y_test, y_prob):.4f}")
        print(f"AUPRC: {average_precision_score(y_test, y_prob):.4f}")


def run_ngram_mlp(task_name: str = "qa") -> None:
    comb = halu_eval_text_pairs(task_name)
    X_train, X_test, y_train, y_test = train_test_split(
        comb["text_to_analyze"], comb["label"], test_size=0.2, random_state=42
    )
    pipe = Pipeline(
        [
            ("ngrams", CountVectorizer(ngram_range=(1, 3), max_df=0.90, min_df=3)),
            ("svd", TruncatedSVD(n_components=100, random_state=42)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    print(f"\n--- N-Gram+SVD+MLP | HaluEval {task_name} ---")
    print(classification_report(y_test, y_pred, target_names=["Factual (0)", "Hallucination (1)"]))
    print(f"AUROC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"AUPRC: {average_precision_score(y_test, y_prob):.4f}")
