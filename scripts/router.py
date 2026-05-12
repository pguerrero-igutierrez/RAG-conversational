"""
router.py
---------
Retrieval Decision Module (Query Router).

Trains a few-shot SetFit binary classifier to decide whether a user query
requires external knowledge retrieval (label=1) or can be answered directly
from the LLM's parametric knowledge (label=0).

Strategy
--------
1. A small seed dataset of labelled Spanish queries is defined inline.
2. SetFit fine-tunes a sentence-transformer on contrastive pairs sampled
   from those seeds (no large annotated corpus required).
3. The trained model is persisted to disk so main.py can reload it cheaply.
4. A rule-based fallback is provided for CPU-only / no-GPU environments
   where SetFit training would be prohibitively slow.

Labels
------
  0 → No retrieval needed  (greetings, arithmetic, common-sense)
  1 → Retrieval needed     (factual, entity-centric, time-sensitive)
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Tuple


BASE_DIR = "./rag_project"
DATA_PATH = f"{BASE_DIR}/data/router_data.json"
MODEL_DIR = f"{BASE_DIR}/router_model"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

if Path(DATA_PATH).exists():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        _raw_data = json.load(f)
    SEED_QUERIES = [(item["text"], item["label"]) for item in _raw_data["train"]]
    EVAL_QUERIES = [(item["text"], item["label"]) for item in _raw_data["test"]]
else:
    SEED_QUERIES = []
    EVAL_QUERIES = []

RETRIEVAL_TRIGGERS = [
    "cuándo", "quién", "qué es", "cuál es", "dónde", "cómo se llama",
    "cuántos", "cuántas", "en qué año", "qué países", "qué idiomas", "qué significa"
]

CONVERSATIONAL_PATTERNS = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "hasta luego",
    "gracias", "por favor", "adiós", "escribe", "redacta", "traduce",
    "resume", "cuánto es", "raíz cuadrada", "más", "menos", "multiplica",
    "chiste", "poema", "correo", "carta"
]

def rule_based_router(query: str) -> int:
    q = query.lower()
    for pat in CONVERSATIONAL_PATTERNS:
        if pat in q:
            return 0
    for pat in RETRIEVAL_TRIGGERS:
        if pat in q:
            return 1
    return 1

def train_setfit_router(
    seed_data: List[Tuple[str, int]] = SEED_QUERIES,
    eval_data: List[Tuple[str, int]] = EVAL_QUERIES,
    base_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    num_epochs: int = 4,
    save_dir: str = MODEL_DIR,
) -> None:

    try:
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("SetFit o datasets no instalados.") from e

    if not seed_data:
        raise ValueError(f"No hay datos en {DATA_PATH} para entrenar.")

    train_ds = Dataset.from_dict({
        "text": [q for q, _ in seed_data],
        "label": [l for _, l in seed_data],
    })

    if eval_data:
        eval_ds = Dataset.from_dict({
            "text": [q for q, _ in eval_data],
            "label": [l for _, l in eval_data],
        })
    else:
        split = train_ds.train_test_split(test_size=0.2, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]

    print(f"[Router] Loading model: {base_model}")
    model = SetFitModel.from_pretrained(base_model)

    args = TrainingArguments(
        output_dir=f"{MODEL_DIR}/checkpoints",
        batch_size=16,
        num_epochs=num_epochs,
        num_iterations=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        metric="f1",
        column_mapping={"text": "text", "label": "label"},
    )

    t0 = time.time()
    trainer.train()
    
    metrics = trainer.evaluate()
    print(f"[Router] Training complete. Metrics: {metrics}")

    model.save_pretrained(save_dir)
    print(f"[Router] Model saved → {save_dir}")

class QueryRouter:
    def __init__(self, model_dir: str = MODEL_DIR, use_setfit: bool = True, threshold: float = 0.35):
        self._setfit_model = None
        self._use_setfit = False
        self._threshold = threshold

        if use_setfit and Path(model_dir).exists():
            try:
                from setfit import SetFitModel
                self._setfit_model = SetFitModel.from_pretrained(model_dir)
                self._use_setfit = True
                print("[Router] SetFit model loaded.")
            except Exception as exc:
                print(f"[Router] Error loading SetFit: {exc}. Using rules.")
        else:
            print("[Router] No model found. Using rules.")

    def needs_retrieval(self, query: str) -> bool:
        if self._use_setfit and self._setfit_model is not None:
            probs = self._setfit_model.predict_proba([query])
            prob_retrieval = float(probs[0][1])
            return prob_retrieval >= self._threshold
        return rule_based_router(query) == 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    if args.train:
        train_setfit_router()

    if args.query:
        router = QueryRouter()
        decision = router.needs_retrieval(args.query)
        print(f"Query   : {args.query}")
        print(f"Decision: {'RETRIEVE' if decision else 'NO RETRIEVE'}")

    if not args.train and not args.query:
        router = QueryRouter(use_setfit=False)
        print("\n── Rule-based router smoke test ──")
        for query, gold in SEED_QUERIES[:8]:
            pred = int(router.needs_retrieval(query))
            mark = "✓" if pred == gold else "✗"
            print(f"  {mark}  [{gold}→{pred}]  {query}")