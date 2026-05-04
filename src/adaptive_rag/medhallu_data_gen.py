"""Phase 1: build router training CSV from MedHallu + Llama generations + embedding similarity labels."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_rag.config import DEFAULT_EMBEDDER_MODEL, DEFAULT_GENERATOR_MODEL, ensure_dirs


def build_medhallu_router_csv(
    output_path: str | Path | None = None,
    batch_size: int = 16,
    model_id: str | None = None,
    embedder_id: str | None = None,
) -> Path:
    ensure_dirs()
    model_id = model_id or DEFAULT_GENERATOR_MODEL
    embedder_id = embedder_id or DEFAULT_EMBEDDER_MODEL
    from adaptive_rag.config import MEDHALLU_TRAINING_CSV

    if output_path:
        out = Path(output_path)
    else:
        env_csv = os.environ.get("MEDHALLU_CSV")
        out = Path(env_csv) if env_csv else MEDHALLU_TRAINING_CSV
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    print("Loading MedHallu (pqa_artificial train)...")
    dataset = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial", split="train")
    df = pd.DataFrame(dataset)

    prompts: list[str] = []
    for _, row in df.iterrows():
        ctx = row["Knowledge"]
        if not isinstance(ctx, str):
            ctx = " ".join(ctx) if isinstance(ctx, (list, tuple)) else str(ctx)
        prompts.append(
            f"Context: {ctx}\n\nQuestion: {row['Question']}\n\nProvide a concise medical answer:\n"
        )
    df["prompt"] = prompts

    generated: list[str] = []
    device = model.device
    for i in tqdm(range(0, len(df), batch_size), desc="generate answers"):
        batch = df["prompt"].iloc[i : i + batch_size].tolist()
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        for j, seq in enumerate(gen_out):
            inp_len = inputs["input_ids"][j].shape[0]
            new_toks = seq[inp_len:]
            generated.append(tokenizer.decode(new_toks, skip_special_tokens=True).strip())
    df["generated_answer"] = generated

    del model
    torch.cuda.empty_cache()

    st_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    grader = SentenceTransformer(embedder_id, device=st_device)
    emb_gen = grader.encode(df["generated_answer"].tolist(), batch_size=256, convert_to_tensor=True)
    emb_gt = grader.encode(df["Ground Truth"].tolist(), batch_size=256, convert_to_tensor=True)
    emb_fake = grader.encode(df["Hallucinated Answer"].tolist(), batch_size=256, convert_to_tensor=True)

    sim_gt = torch.nn.functional.cosine_similarity(emb_gen, emb_gt, dim=1)
    sim_fake = torch.nn.functional.cosine_similarity(emb_gen, emb_fake, dim=1)
    labels = (sim_gt < sim_fake).to(torch.int).cpu().numpy()

    out_df = pd.DataFrame({"prompt": df["prompt"], "label": labels})
    out_df.to_csv(out, index=False)
    uniq, counts = torch.unique(torch.tensor(labels), return_counts=True)
    dist = dict(zip(uniq.tolist(), counts.tolist()))
    print(f"Saved {len(out_df)} rows to {out}")
    print(f"Label distribution (0=safe, 1=hallucination-like): {dist}")
    return out
