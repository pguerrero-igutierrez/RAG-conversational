"""
build_index.py
--------------
Builds the hybrid retrieval index from the SQAC corpus:
  - Tokenizes passages for BM25 (Okapi BM25 with Spanish stemming)
  - Encodes passages with BGE-M3 to produce normalized dense embeddings

Reads
-----
  data/indexes/sqac_corpus.jsonl

Writes
------
  data/indexes/bm25.pkl
  data/indexes/bge_embeddings.npy
  data/indexes/corpus_ids.json

Usage
-----
  python scripts/build_index.py
"""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

import nltk
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

nltk.download("stopwords", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parents[1]
INDEX_DIR       = ROOT_DIR / "data" / "indexes"
CORPUS_PATH     = INDEX_DIR / "sqac_corpus.jsonl"
BM25_PATH       = INDEX_DIR / "bm25.pkl"
EMBEDDINGS_PATH = INDEX_DIR / "bge_embeddings.npy"
CORPUS_IDS_PATH = INDEX_DIR / "corpus_ids.json"

os.makedirs(INDEX_DIR, exist_ok=True)


def log(message: str) -> None:
    print(message, flush=True)

# ── Load corpus ────────────────────────────────────────────────────────────
log(f"[Index] Loading corpus: {CORPUS_PATH}")
with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = [json.loads(l) for l in f]

corpus_texts = [
    f"{r.get('title', '')}: {r['context']}".strip(": ")
    for r in corpus
]
corpus_ids = [r["id"] for r in corpus]
log(f"[Index] {len(corpus):,} passages loaded.")

# ── BM25 ───────────────────────────────────────────────────────────────────
log("\n[Index] Tokenizing for BM25 …")
stemmer    = SnowballStemmer("spanish")
stop_words = set(stopwords.words("spanish"))


def tokenize_for_bm25(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    return [stemmer.stem(w) for w in words if w not in stop_words]


tokenized = [
    tokenize_for_bm25(t)
    for t in tqdm(corpus_texts, desc="BM25 tokenization")
]

log("[Index] Building BM25 index …")
bm25 = BM25Okapi(tokenized)
with open(BM25_PATH, "wb") as f:
    pickle.dump(bm25, f)
log(f"[Index] BM25 index saved → {BM25_PATH}")

# ── BGE-M3 dense embeddings ────────────────────────────────────────────────
log(f"\n[Index] Loading BGE-M3 model …")
bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

log(f"[Index] Encoding {len(corpus_texts):,} passages …")
output = bge_model.encode(
    corpus_texts,
    batch_size=32,
    max_length=512,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
)

embeddings = output["dense_vecs"].astype("float32")
norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / (norms + 1e-10)

np.save(EMBEDDINGS_PATH, embeddings)
log(f"[Index] Dense embeddings saved → {EMBEDDINGS_PATH}  "
    f"shape={embeddings.shape}")

with open(CORPUS_IDS_PATH, "w") as f:
    json.dump(corpus_ids, f)
log(f"[Index] Corpus IDs saved       → {CORPUS_IDS_PATH}")

log("\n[Index] Index build complete.")
log(f"  Passages : {len(corpus):,}")
log(f"  BM25     : {BM25_PATH}")
log(f"  BGE-M3   : {EMBEDDINGS_PATH}")
log(f"  IDs      : {CORPUS_IDS_PATH}")
