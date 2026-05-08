"""
Implements a hybrid retriever combining BM25 lexical scores and bge-m3 dense
cosine similarity, followed by cross-encoder reranking. Exposes a single
retrieve() function to be imported by downstream pipeline modules.
"""

import json
import pickle
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

base_dir = "./rag_project"
index_dir = f"{base_dir}/indexes"

bm25_path = f"{index_dir}/bm25.pkl"
embeddings_path = f"{index_dir}/bge_embeddings.npy"
corpus_ids_path = f"{index_dir}/corpus_ids.json"
corpus_path = f"{index_dir}/sqac_corpus.jsonl"

with open(bm25_path, "rb") as f:
    bm25 = pickle.load(f)

dense_embeddings = np.load(embeddings_path)

with open(corpus_ids_path) as f:
    corpus_ids = json.load(f)

with open(corpus_path, encoding="utf-8") as f:
    corpus = [json.loads(l) for l in f]

id_to_doc = {r["id"]: r for r in corpus}

bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve(question, top_k=10, bm25_weight=0.4, dense_weight=0.6, rerank_top=3):
    tokenized_q = question.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_q))
    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-10)

    q_emb = bge_model.encode(
        [question],
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"].astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    dense_scores = (dense_embeddings @ q_emb.T).squeeze()

    hybrid_scores = bm25_weight * bm25_norm + dense_weight * dense_scores
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    candidates = [id_to_doc[corpus_ids[i]] for i in top_indices]
    pairs = [[question, f"{c['title']}: {c['context']}"] for c in candidates]
    ce_scores = cross_encoder.predict(pairs)

    reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:rerank_top]]


if __name__ == "__main__":
    sample_question = "¿Cuándo se fundó la Universidad de Salamanca?"
    results = retrieve(sample_question)
    print(f"Query: {sample_question}\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc['id']} — {doc['title']}")
        print(f"     {doc['context'][:200]}...\n")
