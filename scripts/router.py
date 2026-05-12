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

import os
import json
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = "./rag_project"
MODEL_DIR  = f"{BASE_DIR}/router_model"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# ── Seed data ──────────────────────────────────────────────────────────────
# Keep balanced: equal positives and negatives so SetFit trains stably.
SEED_QUERIES = [
    # label 1 — retrieval needed (factual / entity-centric)
    ("¿Cuándo se fundó la Universidad de Salamanca?",          1),
    ("¿Quién escribió el Quijote?",                            1),
    ("¿En qué año comenzó la Guerra Civil Española?",          1),
    ("¿Cuál es la capital de Australia?",                      1),
    ("¿Qué es el síndrome de Marfan?",                         1),
    ("¿Cuántos habitantes tiene Ciudad de México?",            1),
    ("¿Quién ganó el Premio Nobel de Literatura en 2023?",     1),
    ("¿Cuándo murió Frida Kahlo?",                             1),
    ("¿Qué países forman la Unión Europea?",                   1),
    ("¿Cuál es la montaña más alta del mundo?",                1),
    ("¿Qué es la fotosíntesis?",                               1),
    ("¿Cuándo se promulgó la Constitución Española de 1978?",  1),
    ("¿Quién fue el primer presidente de Argentina?",          1),
    ("¿Cuál es la velocidad de la luz?",                       1),
    ("¿Qué idiomas se hablan en Suiza?",                       1),
    # label 0 — no retrieval needed (conversational / trivial)
    ("Hola, ¿cómo estás?",                                     0),
    ("¿Puedes ayudarme?",                                      0),
    ("¿Cuánto es 15 más 27?",                                  0),
    ("Buenas tardes.",                                         0),
    ("¿Qué hora es?",                                          0),
    ("Gracias por tu ayuda.",                                  0),
    ("¿Me puedes contar un chiste?",                           0),
    ("Necesito escribir un correo formal.",                    0),
    ("¿Puedes traducir 'hello' al español?",                   0),
    ("Escribe un poema corto sobre el otoño.",                 0),
    ("¿Cuál es la raíz cuadrada de 144?",                      0),
    ("Hasta luego.",                                           0),
    ("¿Cómo se dice 'thank you' en francés?",                  0),
    ("Resume este texto en dos frases.",                       0),
    ("¿Qué significa la palabra 'efímero'?",                   0),
]

# ── Rule-based fallback ────────────────────────────────────────────────────
RETRIEVAL_TRIGGERS = [
    "cuándo", "quién", "qué es", "cuál es", "dónde", "cómo se llama",
    "cuántos", "cuántas", "en qué año", "qué países", "qué idiomas",
    "qué significa",  # ambiguous — lean toward retrieve for definitions
]
CONVERSATIONAL_PATTERNS = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "hasta luego",
    "gracias", "por favor", "adiós", "escribe", "redacta", "traduce",
    "resume", "cuánto es", "raíz cuadrada", "más", "menos", "multiplica",
    "chiste", "poema", "correo", "carta",
]

def rule_based_router(query: str) -> int:
    """
    Lightweight heuristic router used as fallback when SetFit is unavailable.

    Returns 1 (retrieve) or 0 (do not retrieve).
    """
    q = query.lower()
    for pat in CONVERSATIONAL_PATTERNS:
        if pat in q:
            return 0
    for pat in RETRIEVAL_TRIGGERS:
        if pat in q:
            return 1
    # Default: retrieve (safe fallback — better to over-retrieve than miss facts)
    return 1


# ── SetFit trainer ─────────────────────────────────────────────────────────
def train_setfit_router(
    seed_data: list[tuple[str, int]] = SEED_QUERIES,
    base_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    num_epochs: int = 1,
    num_iterations: int = 20,
    save_dir: str = MODEL_DIR,
) -> None:
    """
    Fine-tunes a SetFit model on the seed dataset and saves it to *save_dir*.

    Parameters
    ----------
    seed_data    : list of (query, label) tuples.
    base_model   : HuggingFace sentence-transformer checkpoint to start from.
                   The multilingual MiniLM is fast and supports Spanish well.
    num_epochs   : training epochs for the classification head.
    num_iterations: contrastive pair sampling iterations (SetFit hyperparameter).
    save_dir     : directory where the trained model is persisted.
    """
    try:
        from setfit import SetFitModel, SetFitTrainer
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "SetFit or datasets not installed. "
            "Run: pip install setfit datasets"
        ) from e

    texts  = [q for q, _ in seed_data]
    labels = [l for _, l in seed_data]

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    # SetFit works with very few samples — no train/test split needed here;
    # evaluation is handled separately in evaluate.py.
    train_ds = dataset

    print(f"[Router] Loading base model: {base_model}")
    model = SetFitModel.from_pretrained(base_model)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        num_iterations=num_iterations,
        num_epochs=num_epochs,
        column_mapping={"text": "text", "label": "label"},
    )

    print(f"[Router] Training SetFit on {len(texts)} seed examples …")
    t0 = time.time()
    trainer.train()
    print(f"[Router] Training complete in {time.time() - t0:.1f}s")

    model.save_pretrained(save_dir)
    print(f"[Router] Model saved → {save_dir}")


# ── Inference wrapper ──────────────────────────────────────────────────────
class QueryRouter:
    """
    Wraps either a trained SetFit model or the rule-based fallback.

    Usage
    -----
        router = QueryRouter()
        decision = router.needs_retrieval("¿Quién fue Cervantes?")
        # True  → run retriever
        # False → go straight to LLM
    """

    def __init__(self, model_dir: str = MODEL_DIR, use_setfit: bool = True):
        self._setfit_model = None
        self._use_setfit   = False

        if use_setfit and Path(model_dir).exists():
            try:
                from setfit import SetFitModel
                print(f"[Router] Loading SetFit model from {model_dir} …")
                self._setfit_model = SetFitModel.from_pretrained(model_dir)
                self._use_setfit   = True
                print("[Router] SetFit model loaded.")
            except Exception as exc:
                print(f"[Router] Could not load SetFit model ({exc}). "
                      "Falling back to rule-based router.")
        else:
            print("[Router] No trained model found. Using rule-based router.")

    def needs_retrieval(self, query: str) -> bool:
        """Return True if external retrieval is needed for *query*."""
        if self._use_setfit and self._setfit_model is not None:
            pred = self._setfit_model.predict([query])
            # SetFit returns a tensor or numpy array
            label = int(pred[0])
        else:
            label = rule_based_router(query)
        return label == 1

    def predict_batch(self, queries: list[str]) -> list[int]:
        """Return a list of binary labels (0/1) for a batch of queries."""
        if self._use_setfit and self._setfit_model is not None:
            preds = self._setfit_model.predict(queries)
            return [int(p) for p in preds]
        return [rule_based_router(q) for q in queries]


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test the Query Router.")
    parser.add_argument(
        "--train", action="store_true",
        help="Train and save the SetFit router model."
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query to classify after (optionally) loading the model."
    )
    args = parser.parse_args()

    if args.train:
        train_setfit_router()

    if args.query:
        router = QueryRouter()
        decision = router.needs_retrieval(args.query)
        print(f"Query   : {args.query}")
        print(f"Decision: {'RETRIEVE' if decision else 'NO RETRIEVE'}")

    if not args.train and not args.query:
        # Quick smoke-test on seed data
        router = QueryRouter(use_setfit=False)
        print("\n── Rule-based router smoke test ──")
        for query, gold in SEED_QUERIES[:8]:
            pred = int(router.needs_retrieval(query))
            mark = "✓" if pred == gold else "✗"
            print(f"  {mark}  [{gold}→{pred}]  {query}")
