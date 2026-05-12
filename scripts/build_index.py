"""
Builds the hybrid retrieval index from the corpus: tokenizes passages
for BM25 and encodes them with bge-m3 to produce normalized dense embeddings.
"""

import json
import os
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm.auto import tqdm
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi


nltk.download('stopwords', quiet=True)

base_dir = "./rag_project"
index_dir = f"{base_dir}/indexes"
os.makedirs(index_dir, exist_ok=True)

corpus_path = f"{index_dir}/sqac_corpus.jsonl"
bm25_path = f"{index_dir}/bm25.pkl"
embeddings_path = f"{index_dir}/bge_embeddings.npy"
corpus_ids_path = f"{index_dir}/corpus_ids.json"

with open(corpus_path, encoding="utf-8") as f:
    corpus = [json.loads(l) for l in f]

corpus_texts = [f"{r['title']}: {r['context']}" for r in corpus]
corpus_ids = [r["id"] for r in corpus]

print("Tokenizing for BM25...")
stemmer = SnowballStemmer('spanish')
stop_words = set(stopwords.words('spanish'))

def tokenize_for_bm25(text):
    words = re.findall(r'\w+', text.lower())
    return [stemmer.stem(w) for w in words if w not in stop_words]

tokenized = [tokenize_for_bm25(t) for t in tqdm(corpus_texts, desc="BM25 Tokenization")]

print("Building BM25 index...")
bm25 = BM25Okapi(tokenized)

with open(bm25_path, "wb") as f:
    pickle.dump(bm25, f)

print(f"BM25 index saved → {bm25_path}")

print("Loading bge-m3 model...")
bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

print(f"Encoding {len(corpus_texts)} passages...")
output = bge_model.encode(
    corpus_texts,
    batch_size=32,
    max_length=512,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False
)
embeddings = output["dense_vecs"].astype("float32")
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / (norms + 1e-10)

np.save(embeddings_path, embeddings)

with open(corpus_ids_path, "w") as f:
    json.dump(corpus_ids, f)

print(f"Dense embeddings saved → {embeddings_path}  shape={embeddings.shape}")
print(f"Corpus IDs saved       → {corpus_ids_path}")