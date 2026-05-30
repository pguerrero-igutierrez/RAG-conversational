# Spanish conversational system with router-controlled RAG

[![Poster](https://img.shields.io/badge/Poster-PDF-red)](poster/final_poster.pdf)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/pguerrero-igutierrez/rag-routers-spanish)

**Iker Gutierrez Fandiño & Paula Guerrero Castelló**<br>
University of the Basque Country (EHU) · Natural Language Applications II · 2025–2026

---


This repository provides a Spanish conversational system with a Retrieval-Augmented Generation (RAG) router that decides when a conversational assistant should retrieve external context and when it should answer directly.

The project compares three retrieval policies:

- `always_rag` (baseline): always retrieve from the SQAC corpus before generation.
- `never_rag` (baseline): answer directly with the LLM.
- `router_rag`: use a learned query router to decide whether retrieval is needed.


## Repository Layout

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
├── slurm/                     # Cluster job scripts for the full workflow
├── requirements.txt
└── README.md
```

## Main Components

- `scripts/load_sqac.py`: prepares SQAC retrieval corpus, router splits, and test split.
- `scripts/load_chatsubs.py` and `scripts/load_microsoft_chitchat.py`: prepare non-retrieval conversational examples.
- `scripts/build_index.py`: builds BM25 and BGE-M3 dense retrieval indexes.
- `scripts/retriever.py`: runs hybrid retrieval and reranking.
- `scripts/router.py`: trains/evaluates router approaches on dataset labels: `frozen_lr`, `setfit`, and `finetune`.
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

Alternatively, the code can read `HF_TOKEN` from a local `config.py`.


## Running the System

### Quick Test: Single Query

```bash
python scripts/main.py \
  --mode single \
  --strategy router_rag \
  --router_approach setfit \
  --router_size 15036 \
  --query "¿Quién escribió Don Quijote?"
```

### Full Reproducible Workflow

The following steps reproduce the complete system and experiments reported in the results table. Execute in order:

#### 1. Data Preparation

```bash
# Load SQAC questions and prepare train/dev/test splits
python scripts/load_sqac.py

# Load conversational examples (non-retrieval)
python scripts/load_chatsubs.py
```

#### 2. Build Retrieval Index

```bash
# Build BM25 and BGE-M3 dense retrieval indexes
python scripts/build_index.py
```

#### 3. Train Router Models

Train three router approaches on the full SQAC training set:

```bash
# Train frozen_lr router
python scripts/router.py --train --approach frozen_lr

# Train SetFit router
python scripts/router.py --train --approach setfit

# Train fine-tune router
python scripts/router.py --train --approach finetune
```

#### 4. Run Batch Predictions (Baseline Strategies)

```bash
# Baseline: Always retrieve from SQAC
python scripts/main.py --mode batch --strategy always_rag

# Baseline: Never retrieve, LLM only
python scripts/main.py --mode batch --strategy never_rag
```

#### 5. Run Batch Predictions (Router-Controlled RAG)

```bash
# Router strategy: frozen_lr approach (size 15036 per class)
python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach frozen_lr \
  --router_size 15036

# Router strategy: SetFit approach (size 15036 per class)
python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach setfit \
  --router_size 15036

# Router strategy: Fine-tune approach (size 12500 per class)
python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach finetune \
  --router_size 12500
```

#### 6. Oracle-Labeled Router Training (Optional)

Build oracle-labeled training sets using uncertainty sampling:

```bash
# Build oracle training set: 64 examples per label
python scripts/build_oracle_train_subset.py \
  --target_per_label 64 \
  --candidate_batch 8 \
  --seed 42

# Build oracle training set: 500 examples per label
python scripts/build_oracle_train_subset.py \
  --target_per_label 500 \
  --candidate_batch 8 \
  --seed 42 \
  --output corpus/oracle_train_sqac_500_per_label.jsonl
```

Train SetFit routers on oracle-labeled sets:

```bash
# Train SetFit on 64-per-label oracle set
python scripts/train_oracle_setfit.py \
  --data corpus/oracle_train_sqac_64_per_label.jsonl \
  --save_dir models/router_oracle/setfit/64_per_label \
  --num_epochs 4 \
  --num_iterations 24 \
  --batch_size 16

# Train SetFit on 500-per-label oracle set
python scripts/train_oracle_setfit.py \
  --data corpus/oracle_train_sqac_500_per_label.jsonl \
  --save_dir models/router_oracle/setfit/500_per_label \
  --num_epochs 4 \
  --num_iterations 24 \
  --batch_size 16
```

Run predictions with oracle SetFit routers:

```bash
# Predictions with oracle SetFit (64 per label)
python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach setfit \
  --router_model_dir models/router_oracle/setfit/64_per_label

# Predictions with oracle SetFit (500 per label)
python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach setfit \
  --router_model_dir models/router_oracle/setfit/500_per_label
```

#### 7. Latency Measurements (Without Self-Feedback)

```bash
# Measure latency with no feedback (always_rag baseline)
python scripts/main.py \
  --mode batch \
  --strategy always_rag \
  --no_feedback \
  --run_name latency_nofb_always_rag

# Measure latency with no feedback (never_rag baseline)
python scripts/main.py \
  --mode batch \
  --strategy never_rag \
  --no_feedback \
  --run_name latency_nofb_never_rag

# Measure latency with no feedback (all router variants)
python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach frozen_lr \
  --router_size 15036 \
  --no_feedback \
  --run_name latency_nofb_router_frozen_lr

python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach setfit \
  --router_size 15036 \
  --no_feedback \
  --run_name latency_nofb_router_setfit

python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach finetune \
  --router_size 12500 \
  --no_feedback \
  --run_name latency_nofb_router_finetune
```

#### 8. Final Evaluation

Evaluate all strategies and generate the report:

```bash
# Evaluate all strategies (tracked predictions + newly generated ones)
python scripts/evaluate.py --strategy all

# Verbose evaluation report
python scripts/evaluate.py --strategy all --verbose
```

Results are saved to `outputs/evaluation_report.json`.

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

## Reproducibility Notes

- `outputs/` contains the saved prediction and evaluation artifacts used to reproduce the reported results.
- `corpus/oracle_train_sqac_*.jsonl` contains derived oracle-labeled SQAC examples used for router training experiments.


---

## Contact

**Iker Gutierrez Fandiño**<br>
*Computational linguist*<br>
[GitHub Profile](https://github.com/iker-gutierrez) | [LinkedIn](https://www.linkedin.com/in/iker-gutierrez-fandino)

**Paula Guerrero Castelló**<br> 
*Computational linguist & translator*<br>
[GitHub Profile](https://github.com/guerreropaula) | [LinkedIn](https://www.linkedin.com/in/paula-guerrero-castelló)
