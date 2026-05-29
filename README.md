# Spanish conversational system with router-controlled RAG

[![Poster](https://img.shields.io/badge/Poster-PDF-red)](poster/final_poster.pdf)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/pguerrero-igutierrez/rag-routers-spanish)

**Iker Gutierrez Fandiño & Paula Guerrero Castelló**<br>
University of the Basque Country (EHU) · Natural Language Applications II · 2025–2026

---

Spanish Retrieval-Augmented Generation (RAG) experiments for deciding when a conversational assistant should retrieve external context and when it should answer directly.

The project compares three retrieval policies:

- `always_rag`: always retrieve from the SQAC corpus before generation.
- `never_rag`: answer directly with the LLM.
- `router_rag`: use a learned query router to decide whether retrieval is needed.

The pipeline combines SQAC question answering data, conversational/chitchat data, hybrid retrieval, Mixtral generation, self-feedback, and several learned router variants.

## Repository Layout

```text
.
├── corpus/                    # Tracked source/evaluation corpora
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   ├── oracle_train_sqac_64_per_label.jsonl
│   └── oracle_train_sqac_500_per_label.jsonl
├── outputs/                   # Tracked final reports and prediction files
├── scripts/                   # Data prep, indexing, routing, generation, eval
├── slurm/                     # Cluster job scripts for the full workflow
├── requirements.txt
└── README.md
```

Ignored local artifacts include `data/`, `models/`, `logs/`, `hf_cache/`, `past/`, Python caches, local `config.py`, and poster build/source files such as `poster/template.tex`.

## Main Components

- `scripts/load_sqac.py`: prepares SQAC retrieval corpus, router splits, and test split.
- `scripts/load_chatsubs.py` and `scripts/load_microsoft_chitchat.py`: prepare non-retrieval conversational examples.
- `scripts/build_index.py`: builds BM25 and BGE-M3 dense retrieval indexes.
- `scripts/retriever.py`: runs hybrid retrieval and reranking.
- `scripts/router.py`: trains/evaluates router approaches: `frozen_lr`, `setfit`, and `finetune`.
- `scripts/train_oracle_setfit.py`: trains SetFit routers on oracle-labeled corpora in `corpus/`.
- `scripts/generator.py`: handles prompt construction, generation, and self-feedback.
- `scripts/main.py`: runs interactive, single-query, or batch prediction modes.
- `scripts/evaluate.py`: computes retrieval decision, response quality, retrieval quality, oracle policy, feedback, and latency metrics.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The generator uses `mistralai/Mixtral-8x7B-Instruct-v0.1`. If your Hugging Face account requires authentication, set:

```bash
export HF_TOKEN=your_token_here
```

Alternatively, the code can read `HF_TOKEN` from a local `config.py`; this file is intentionally ignored.

## Data and Generated Artifacts

Tracked:

- Raw SQAC JSON files in `corpus/`
- Oracle-labeled training corpora in `corpus/`
- Final prediction files and `outputs/evaluation_report.json`

Ignored and regenerated locally:

- `data/processed/`
- `data/indexes/`
- `models/`
- `logs/`
- Hugging Face/model caches

Build the processed data and retrieval index:

```bash
python scripts/load_sqac.py
python scripts/load_chatsubs.py
python scripts/build_index.py
```

## Running the System

Single query:

```bash
python scripts/main.py \
  --mode single \
  --strategy router_rag \
  --router_approach setfit \
  --router_size 15036 \
  --query "¿Quién escribió Don Quijote?"
```

Interactive mode:

```bash
python scripts/main.py --mode interactive --strategy router_rag
```

Batch predictions:

```bash
python scripts/main.py --mode batch --strategy always_rag
python scripts/main.py --mode batch --strategy never_rag
python scripts/main.py --mode batch --strategy router_rag --router_approach setfit --router_size 15036
```

Evaluate tracked or newly generated predictions:

```bash
python scripts/evaluate.py
python scripts/evaluate.py --verbose
```

## Slurm Workflow

The `slurm/` directory contains the cluster workflow used for the experiments:

```text
01_load_sqac.sh
02_load_chatsubs.sh
03_build_index.sh
04_train_router.sh
05_batch_always_rag.sh
06_batch_never_rag.sh
07_batch_router_rag.sh
08_evaluate.sh
09_build_oracle_train_subset.sh
10_train_oracle_setfit.sh
11_batch_oracle_setfit.sh
12_build_oracle_train_500.sh
13_train_oracle_setfit_500.sh
14_batch_oracle_setfit_500.sh
```

Submit jobs with `sbatch`, for example:

```bash
sbatch slurm/03_build_index.sh
sbatch slurm/04_train_router.sh
sbatch slurm/07_batch_router_rag.sh
sbatch slurm/08_evaluate.sh
```

The Slurm scripts assume the project lives at `/home/igutierrez134/apps2` and use the environment at `/home/igutierrez134/envs/apps2_3.11`.

## Results Snapshot

The tracked report in `outputs/evaluation_report.json` evaluates 170 total samples: 85 SQAC questions and 85 conversational prompts.

| Strategy | Retrieve NoSF/SF | Latency NoSF/SF | Token F1 | BERTScore F1 | Dataset acc. NoSF/SF | Oracle acc. NoSF/SF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_rag` | 100.00 / 61.76 | 6.21s / 13.94s | 7.88 | 61.10 | 50.00 / 61.18 | 69.41 / 77.65 |
| `never_rag` | 0.00 / 0.00 | 6.68s / 6.64s | 4.57 | 58.21 | 50.00 / 50.00 | 30.59 / 30.59 |
| `router_rag_frozen_lr_15036` | 48.24 / 34.12 | 5.82s / 10.33s | 7.39 | 60.81 | 97.06 / 84.12 | 67.06 / 72.94 |
| `router_rag_setfit_15036` | 48.82 / 35.88 | 5.54s / 10.05s | 8.12 | 61.22 | 98.82 / 85.88 | 69.41 / 76.47 |
| `router_rag_finetune_12500` | 50.59 / 37.65 | 5.53s / 9.98s | 7.64 | 61.09 | 99.41 / 87.65 | 69.41 / 75.29 |
| `router_rag_oracle_setfit_500` | 47.65 / 32.35 | 6.77s / 9.82s | 6.98 | 59.78 | 52.94 / 58.82 | 57.65 / 57.65 |
| `router_rag_oracle_setfit_64` | 44.12 / 22.94 | 6.75s / 10.10s | 6.03 | 59.68 | 52.94 / 57.65 | 49.41 / 49.41 |

NoSF latency comes from the no-feedback rerun summarized in `outputs/latency_pre_post_summary.json`. SF metrics come from the saved full self-feedback runs in `outputs/evaluation_report.json`.

## Notes

- `outputs/` is tracked in this repository because the prediction files are small and useful for reproducing the reported evaluation.
- `corpus/oracle_train_sqac_*.jsonl` contains derived oracle-labeled SQAC training examples and is tracked as project data.
- Poster files are not part of the GitHub-facing project state; `poster/template.tex` is ignored intentionally.
