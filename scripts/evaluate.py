"""
08_evaluate.py

Evaluates the RAG pipeline against the SQAC validation set.
Computes:
- Retrieval: Recall@k, MRR
- Generation: Token F1, BERTScore
- Strategies: Always RAG, Never RAG, Router RAG, Oracle.
"""

import json
import numpy as np
from tqdm.auto import tqdm
from bert_score import score as bert_score
import importlib
import collections

pipeline_module = importlib.import_module("pipeline")
retriever_module = importlib.import_module("retriever")

EVAL_FILE = "./rag_project/data/sqac_validation.jsonl"