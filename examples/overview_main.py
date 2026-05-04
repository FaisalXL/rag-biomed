#!/usr/bin/env python3
"""Portfolio overview (no GPU). Paired with `notebooks/overview.ipynb`. Run: `python examples/overview_main.py`"""

from __future__ import annotations

OVERVIEW = """
Biomedical Adaptive RAG — portfolio overview
==============================================

Idea
----
A router scores each medical question for hallucination / grounding risk. Low risk:
answer with Llama from memory. High risk: retrieve PubMed abstracts (FAISS) and
condition the answer on context.

Code layout
-----------
  src/adaptive_rag/     Library code (data gen, FAISS, LoRA, shootout, reports)
  scripts/00–06.py      Full pipeline CLIs (reproduce paper-scale runs)
  notebooks/            Jupyter walkthroughs (same flows as examples/*.py)
  examples/             Runnable Python twins of the notebooks (no Jupyter)

Typical pipeline (GPU + HF_TOKEN for Llama)
------------------------------------------
  export PYTHONPATH="$(pwd)/src"
  python scripts/02_generate_medhallu_router_csv.py
  python scripts/03_train_lora_router.py
  python scripts/04_build_pubmed_faiss.py
  python scripts/05_train_baseline_routers.py
  python scripts/06_adaptive_rag_shootout.py --n-samples 100

Quick interactive demo (fewer questions)
------------------------------------------
  python examples/adaptive_rag_demo.py --n-samples 25

See README.md for results table, architecture, and course context.
"""


def main() -> None:
    print(OVERVIEW.strip())


if __name__ == "__main__":
    main()
