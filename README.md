# Spanish conversational system with router-controlled RAG

[![Poster](https://img.shields.io/badge/Poster-PDF-red)](poster.pdf)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/pguerrero-igutierrez/rag-routers-spanish)

**Iker Gutierrez Fandiño & Paula Guerrero Castelló**<br>
University of the Basque Country (EHU) · Natural Language Applications II · 2025–2026

---


This repository provides a Spanish conversational system with a Retrieval-Augmented Generation (RAG) router that decides when a conversational assistant should retrieve external context and when it should answer directly.

The project compares three retrieval policies:

- `always_rag` (baseline): always retrieve from the SQAC corpus before generation.
- `never_rag` (baseline): answer directly with the LLM.
- `router_rag`: use a learned query router to decide whether retrieval is needed.

The pipeline combines hybrid retrieval (BM25 + BGE-M3 + MiniLM reranker) with a self-feedback loop inspired by Self-RAG, and evaluates three router training strategies across eleven data sizes.

## Repository layout

```text
.
├── corpus/                    # Source/evaluation corpora
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   ├── oracle_train_sqac_64_per_label.jsonl # Oracle-labeled training corpora (128 samples)
│   └── oracle_train_sqac_500_per_label.jsonl # Oracle-labeled training corpora (1000 samples)
├── outputs/                   # Final evaluation reports and prediction files
├── scripts/                   # Data prep, indexing, routing, generation, eval
├── poster.pdf                 
├── requirements.txt
└── README.md
```

### Scripts overview
 
| Script | Purpose |
|---|---|
| `load_sqac.py` | Parse SQAC JSON → corpus, router train/val splits, test split |
| `load_chatsubs.py` | Extract Spanish chitchat turns from ChatSubs (router train + val) |
| `load_microsoft_chitchat.py` | Build held-out chitchat test set from MS QnA Maker |
| `build_index.py` | Build BM25 index and BGE-M3 dense embeddings from corpus |
| `retriever.py` | Hybrid retrieval (BM25 + BGE-M3) with MiniLM cross-encoder reranking |
| `router.py` | Train `frozen_lr`, `setfit`, and `finetune` routers; learning curve |
| `train_oracle_setfit.py` | Train SetFit on oracle-labelled data |
| `build_oracle_train_subset.py` | Generate oracle labels by comparing always/never RAG via BERTScore |
| `generator.py` | Prompt construction, Mixtral generation, self-feedback loop |
| `main.py` | Orchestrator: interactive, single-query, and batch modes |
| `evaluate.py` | Full evaluation suite (routing, quality, retrieval, oracle, latency) |
| `summarize_latency_nofb.py` | Compute pre/post self-feedback latency summary |


## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The generator requires `mistralai/Mixtral-8x7B-Instruct-v0.1`. Set your token if needed:

```bash
export HF_TOKEN=your_token_here
```

Alternatively, the code can read `HF_TOKEN` from a local `config.py`.

---

## Running the system

### Quick test: single query

```bash
python scripts/main.py \
  --mode single \
  --strategy router_rag \
  --router_approach setfit \
  --router_size 15036 \
  --query "¿Quién escribió Don Quijote?"
```

---
## Full workflow
 
Steps 1–8 reproduce all reported results. Steps 6–8 are optional extras.
 
### 1. Data preparation
 
```bash
# SQAC corpus + router train/val splits + test split (85 questions)
python scripts/load_sqac.py
 
# ChatSubs chitchat turns for router train + val
python scripts/load_chatsubs.py
 
# Microsoft QnA Maker held-out chitchat test set (85 prompts)
python scripts/load_microsoft_chitchat.py
```
 
This produces:
 
| File | Samples | Label |
|---|---|---|
| `data/processed/router_train_sqac.jsonl` | 15,036 | 1 |
| `data/processed/router_train_chitchat.jsonl` | 15,036 | 0 |
| `data/processed/router_val_sqac.jsonl` | 1,779 | 1 |
| `data/processed/router_val_chitchat.jsonl` | 1,664 | 0 |
| `data/processed/test_sqac.jsonl` | 85 | 1 |
| `data/processed/test_chitchat.jsonl` | 85 | 0 |
| `data/indexes/sqac_corpus.jsonl` | 5,601 passages | — |
 
### 2. Build retrieval index
 
```bash
python scripts/build_index.py
```
 
Writes `data/indexes/bm25.pkl`, `data/indexes/bge_embeddings.npy`, and `data/indexes/corpus_ids.json`.
 
### 3. Train router models
 
Runs the full learning curve (sizes 8–15,036 per class) for all three approaches:
 
```bash
python scripts/router.py --train --approach all
```
 
Train a single approach or size:
 
```bash
python scripts/router.py --train --approach setfit
python scripts/router.py --train --approach finetune --size 1000
```
 
Checkpoints are saved to `models/router/<approach>/<size>/`. Results go to `models/router/learning_curve.json`.
 
### 4. Baseline batch predictions
 
```bash
python scripts/main.py --mode batch --strategy always_rag
python scripts/main.py --mode batch --strategy never_rag
```
 
### 5. Router-controlled RAG batch predictions
 
```bash
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach frozen_lr --router_size 15036
 
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach setfit --router_size 15036
 
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach finetune --router_size 12500
```
 
Prediction files are written to `outputs/predictions_<run_name>.jsonl`.
 
### 6. Oracle-labelled router training (optional)
 
Build oracle labels by running both always/never RAG on SQAC training examples and scoring with BERTScore:
 
```bash
# 64 examples per label (128 total)
python scripts/build_oracle_train_subset.py \
  --target_per_label 64 --candidate_batch 8 --seed 42
 
# 500 examples per label (1,000 total)
python scripts/build_oracle_train_subset.py \
  --target_per_label 500 --candidate_batch 8 --seed 42 \
  --output corpus/oracle_train_sqac_500_per_label.jsonl
```
 
Train SetFit on oracle-labelled data:
 
```bash
python scripts/train_oracle_setfit.py \
  --data corpus/oracle_train_sqac_64_per_label.jsonl \
  --save_dir models/router_oracle/setfit/64_per_label \
  --num_epochs 4 --num_iterations 24 --batch_size 16
 
python scripts/train_oracle_setfit.py \
  --data corpus/oracle_train_sqac_500_per_label.jsonl \
  --save_dir models/router_oracle/setfit/500_per_label \
  --num_epochs 4 --num_iterations 24 --batch_size 16
```
 
Run predictions with oracle routers:
 
```bash
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach setfit \
  --router_model_dir models/router_oracle/setfit/64_per_label \
  --run_name router_rag_oracle_setfit_64
 
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach setfit \
  --router_model_dir models/router_oracle/setfit/500_per_label \
  --run_name router_rag_oracle_setfit_500
```
 
### 7. Latency measurements without self-feedback (optional)
 
Re-run all strategies with `--no_feedback` to isolate base latency:
 
```bash
python scripts/main.py --mode batch --strategy always_rag \
  --no_feedback --run_name latency_nofb_always_rag
 
python scripts/main.py --mode batch --strategy never_rag \
  --no_feedback --run_name latency_nofb_never_rag
 
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach frozen_lr --router_size 15036 \
  --no_feedback --run_name latency_nofb_router_rag_frozen_lr_15036
 
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach setfit --router_size 15036 \
  --no_feedback --run_name latency_nofb_router_rag_setfit_15036
 
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach finetune --router_size 12500 \
  --no_feedback --run_name latency_nofb_router_rag_finetune_12500
 
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach setfit \
  --router_model_dir models/router_oracle/setfit/500_per_label \
  --no_feedback --run_name latency_nofb_router_rag_oracle_setfit_500
 
python scripts/main.py --mode batch --strategy router_rag \
  --router_approach setfit \
  --router_model_dir models/router_oracle/setfit/64_per_label \
  --no_feedback --run_name latency_nofb_router_rag_oracle_setfit_64
 
# Summarise pre/post self-feedback latency
python scripts/summarize_latency_nofb.py
```
 
### 8. Evaluation
 
```bash
python scripts/evaluate.py --strategy all
python scripts/evaluate.py --strategy all --verbose  # per-sample breakdown
```
 
Results are printed to stdout and saved to `outputs/evaluation_report.json`.
 
---
 
## Interactive and single-query modes
 
```bash
# Interactive REPL
python scripts/main.py --mode interactive --strategy router_rag
 
# Single query
python scripts/main.py --mode single \
  --strategy router_rag \
  --router_approach setfit \
  --router_size 15036 \
  --query "¿Quién escribió Don Quijote?"
```
 
---
 
## Results
 
Evaluated on 170 samples (85 SQAC + 85 chitchat). NoSF = no self-feedback; SF = with self-feedback.
 
| Strategy | Retrieve NoSF/SF | Latency NoSF/SF | Token F1 | BERTScore F1 | Dataset acc. NoSF/SF | Oracle acc. NoSF/SF |
|---|---:|---:|---:|---:|---:|---:|
| `always_rag` | 100.0 / 61.8 | 6.21s / 13.94s | 7.88 | 61.10 | 50.0 / 61.2 | 69.4 / 77.7 |
| `never_rag` | 0.0 / 0.0 | 6.68s / 6.64s | 4.57 | 58.21 | 50.0 / 50.0 | 30.6 / 30.6 |
| `router_rag_frozen_lr_15036` | 48.2 / 34.1 | 5.82s / 10.33s | 7.39 | 60.81 | 97.1 / 84.1 | 67.1 / 72.9 |
| `router_rag_setfit_15036` | 48.8 / 35.9 | 5.54s / 10.05s | **8.12** | **61.22** | 98.8 / 85.9 | 69.4 / 76.5 |
| `router_rag_finetune_12500` | 50.6 / 37.6 | 5.53s / **9.98s** | 7.64 | 61.09 | **99.4** / **87.7** | 69.4 / 75.3 |
| `router_rag_oracle_setfit_500` | 47.6 / 32.4 | 6.77s / 9.82s | 6.98 | 59.78 | 52.9 / 58.8 | 57.7 / 57.7 |
| `router_rag_oracle_setfit_64` | 44.1 / 22.9 | 6.75s / 10.10s | 6.03 | 59.68 | 52.9 / 57.7 | 49.4 / 49.4 |
 
Key findings:
 
1. **A learned router is worthwhile.** SetFit gives the best quality/cost trade-off: slightly better answer quality while reducing retrieval from 100% to 48.8%, a 27.9% latency reduction relative to Always RAG.
2. **SetFit is the practical router choice.** It reaches 97.6% validation macro F1 with only 8 examples per class and remains competitive with full fine-tuning, which only pulls ahead at larger training sizes.
3. **Oracle-label training is harder than dataset-label training.** Routers trained on oracle labels underperform the dataset-label routers on oracle accuracy.
4. **Self-feedback mainly catches unnecessary retrieval.** Dataset-label routers retrieve in about half of the samples before SF, but only 34.1–37.6% after SF, while oracle accuracy improves by 5.88–7.06 points.

---
 
## Reproducibility
 
- `outputs/` contains tracked prediction files and `evaluation_report.json` for all reported results.
- `corpus/oracle_train_sqac_*.jsonl` contains the derived oracle-labelled training sets.
- Regenerated artifacts (`data/processed/`, `data/indexes/`, `models/`) are git-ignored and must be rebuilt locally.
- NoSF latency comes from the no-feedback rerun summarized in `outputs/latency_pre_post_summary.json`. SF metrics come from the saved full self-feedback runs in `outputs/evaluation_report.json`.
---
 
## Contact

**Iker Gutierrez Fandiño**<br>
*Computational linguist*<br>
[GitHub Profile](https://github.com/iker-gutierrez) | [LinkedIn](https://www.linkedin.com/in/iker-gutierrez-fandino)

**Paula Guerrero Castelló**<br> 
*Computational linguist & translator*<br>
[GitHub Profile](https://github.com/guerreropaula) | [LinkedIn](https://www.linkedin.com/in/paula-guerrero-castelló)
