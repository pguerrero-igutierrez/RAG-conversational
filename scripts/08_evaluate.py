"""
08_evaluate.py

Evaluates the RAG pipeline against the SQAC validation set.
Computes:
- Retrieval: Recall@k, MRR
- Generation: Token F1, BERTScore
- Strategies: Always RAG, Never RAG, Router RAG, Oracle.
"""