# Legacy Colab notebooks (archive only)

This directory contains **original Colab exports** from the team: large widget metadata, execution outputs, and exploratory cells. They are **not** imported by `src/`, `scripts/`, `notebooks/`, or `main.py`.

Use them when you need a **historical screenshot**, a figure, or to diff against the refactored code. The maintained execution path is:

- **Library:** `src/adaptive_rag/`
- **Full runs:** `scripts/00`–`06`
- **Portfolio:** `notebooks/pipeline.ipynb` (and repo root `main.py`)

| Legacy file | Topic |
|-------------|--------|
| `internal_state_llama3_1_8B.ipynb` | HaluEval hidden-state probes |
| `meanEntropyvarentropy.ipynb` | Entropy / semantic-stability ablations |
| `ngram_basic.ipynb`, `Brouge.ipynb` | Text-only baselines |
| `dataGen.ipynb` | MedHallu → router CSV |
| `vectorDBbuild.ipynb` | PubMed FAISS corpus |
| `PEFT_lora(adaptiveRAG).ipynb` | LoRA router training |
| `comparing_Routers.ipynb`, `testingLORA.ipynb` | Router bake-off + adaptive benchmark |
| `FAISS_adapter+pipeline.ipynb` | k-NN + entropy prototype router |
