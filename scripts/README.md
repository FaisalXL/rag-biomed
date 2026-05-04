## Pipeline scripts (`scripts/`)

| Step | Command | Purpose |
|------|---------|---------|
| 0 (opt) | `python scripts/00_text_baselines_haluval.py --task qa --method tfidf` | HaluEval text baselines |
| 1 (opt) | `python scripts/01_internal_state_probe.py --tasks qa --max-samples 800` | Hidden-state linear probe (default Qwen; set `PROBE_MODEL_ID` for Llama) |
| 2 | `python scripts/02_generate_medhallu_router_csv.py` | Build `data/medhallu_lora_training_data_batch.csv` |
| 3 | `python scripts/03_train_lora_router.py` | LoRA sequence classifier on Llama |
| 4 | `python scripts/04_build_pubmed_faiss.py` | FAISS index + mapping under `artifacts/` |
| 5 (opt) | `python scripts/05_train_baseline_routers.py` | XGBoost joblib + DistilBERT folder |
| 6 | `python scripts/06_adaptive_rag_shootout.py --n-samples 100` | Shootout only (assumes artifacts already exist) |

`python main.py final` orchestrates **02→05** when needed, then runs the same shootout logic as step **06**.

---
