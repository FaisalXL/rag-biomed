#!/usr/bin/env python3
"""Step 1 (optional motivation): linear probe on last-token hidden states — HaluEval."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import login

from adaptive_rag.internal_state_probe import run_hidden_state_probe


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-id",
        default=os.environ.get("PROBE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"),
        help="Causal LM with output_hidden_states (no gating for Qwen)",
    )
    p.add_argument("--tasks", nargs="*", default=["qa", "dialogue", "summarization"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    run_hidden_state_probe(
        model_id=args.model_id,
        tasks=args.tasks,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
