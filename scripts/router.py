"""
router.py
---------
Retrieval Decision Module (Query Router).

Trains three learned approaches across training set sizes
and validates each with macro F1 on a fixed labelled val set.

Approaches
----------
  frozen_lr  — frozen paraphrase-multilingual-mpnet-base-v2 + logistic
               regression. Trains in seconds. Strong baseline.

  setfit     — SetFit contrastive fine-tuning. Few-shot specialist.

  finetune   — Full encoder fine-tuning via HuggingFace Trainer.
               Expected best performer at large data sizes.

Learning curve sizes (per class, balanced)
------------------------------------------
  8, 64, 128, 250, 500, 1000, 2500, 5000, 7500, 12500, 15036

Validation
----------
  Fixed val set: 3,328 samples (1,664 SQAC label=1 + 1,664 ChatSubs label=0).
  Metric: macro F1 on labelled val set.
  Frequency: once per epoch (finetune/setfit), once total (frozen_lr).
  Best checkpoint per run saved by val F1.

  Oracle routing validation is NOT done here — see router_oracle.py
  for the winning configuration identified from the learning curve.

Outputs
-------
  models/router/frozen_lr/<size>/     best frozen_lr checkpoint per size
  models/router/setfit/<size>/        best setfit checkpoint per size
  models/router/finetune/<size>/      best finetune checkpoint per size
  models/router/learning_curve.json   full results for plotting

Usage
-----
  # Full learning curve, all approaches:
  python scripts/router.py --train --approach all

  # Single approach:
  python scripts/router.py --train --approach finetune

  # Single approach, single size:
  python scripts/router.py --train --approach finetune --size 1000

  # Classify a query with best available model:
  python scripts/router.py --query "¿Quién fue Cervantes?"

  # Rule-based smoke test:
  python scripts/router.py --smoke_test
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR     = ROOT_DIR / "models" / "router"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

ROUTER_TRAIN_SQAC     = PROCESSED_DIR / "router_train_sqac.jsonl"
ROUTER_TRAIN_CHITCHAT = PROCESSED_DIR / "router_train_chitchat.jsonl"
ROUTER_VAL_SQAC       = PROCESSED_DIR / "router_val_sqac.jsonl"
ROUTER_VAL_CHITCHAT   = PROCESSED_DIR / "router_val_chitchat.jsonl"

LEARNING_CURVE_PATH = MODEL_DIR / "learning_curve.json"

# ── Learning curve sizes (per class) ──────────────────────────────────────
LEARNING_CURVE_SIZES = [8, 64, 128, 250, 500, 1000, 2500, 5000, 7500, 12500, 15036]

# ── Encoder ────────────────────────────────────────────────────────────────
ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOKENIZER_KWARGS = {"fix_mistral_regex": True}

# ── Rule-based fallback ────────────────────────────────────────────────────
RETRIEVAL_TRIGGERS = [
    "cuándo", "quién", "qué es", "cuál es", "dónde", "cómo se llama",
    "cuántos", "cuántas", "en qué año", "qué países", "qué idiomas",
    "qué significa",
]
CONVERSATIONAL_PATTERNS = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "hasta luego",
    "gracias", "por favor", "adiós", "escribe", "redacta", "traduce",
    "resume", "cuánto es", "raíz cuadrada", "chiste", "poema", "correo",
    "carta",
]


def rule_based_router(query: str) -> int:
    q = query.lower()
    for pat in RETRIEVAL_TRIGGERS:
        if pat in q:
            return 1
    for pat in CONVERSATIONAL_PATTERNS:
        if pat in q:
            return 0
    return 1


# ── Data loading ───────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_split(
    sqac_path:     Path,
    chitchat_path: Path,
    n_per_class:   Optional[int] = None,
    seed:          int = 42,
) -> tuple[list[str], list[int]]:
    """
    Load and merge SQAC (label=1) and ChatSubs (label=0).
    If n_per_class is set, subsample that many from each class.
    """
    sqac     = _load_jsonl(sqac_path)
    chitchat = _load_jsonl(chitchat_path)

    if n_per_class is not None:
        rng  = random.Random(seed)
        sqac     = rng.sample(sqac,     min(n_per_class, len(sqac)))
        chitchat = rng.sample(chitchat, min(n_per_class, len(chitchat)))

    records = sqac + chitchat
    texts   = [r.get("question", r.get("text", "")) for r in records]
    labels  = [r["label"] for r in records]
    return texts, labels


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(preds: list[int], labels: list[int]) -> dict:
    from sklearn.metrics import f1_score, classification_report
    macro_f1 = f1_score(labels, preds, average="macro")
    f1pc     = f1_score(labels, preds, average=None).tolist()
    report   = classification_report(
        labels, preds,
        target_names=["no_retrieval", "retrieval"],
    )
    return {
        "macro_f1":     round(float(macro_f1), 4),
        "f1_per_class": [round(float(x), 4) for x in f1pc],
        "report":       report,
    }


def print_metrics(label: str, metrics: dict, elapsed: float) -> None:
    print(f"\n── {label} ──────────────────────────────────────────────")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  F1 [no_ret/ret]: {metrics['f1_per_class']}")
    print(f"  Time           : {elapsed:.1f}s")


# ── Model save dir ─────────────────────────────────────────────────────────
def _save_dir(approach: str, size: int) -> str:
    d = MODEL_DIR / approach / str(size)
    Path(d).mkdir(parents=True, exist_ok=True)
    return str(d)


# ══════════════════════════════════════════════════════════════════════════
# Approach 1 — Frozen embeddings + Logistic Regression
# ══════════════════════════════════════════════════════════════════════════
def train_frozen_lr(
    train_texts:  list[str],
    train_labels: list[int],
    val_texts:    list[str],
    val_labels:   list[int],
    size:         int,
) -> dict:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    save_dir = _save_dir("frozen_lr", size)
    t0       = time.time()

    # Load encoder once if possible (expensive); reuse across sizes
    print(f"  [FrozenLR] Encoding {len(train_texts):,} train samples …")
    encoder   = SentenceTransformer(
        ENCODER_MODEL,
        processor_kwargs=TOKENIZER_KWARGS,
    )
    train_emb = encoder.encode(
        train_texts, normalize_embeddings=True, show_progress_bar=False,
    )
    val_emb = encoder.encode(
        val_texts, normalize_embeddings=True, show_progress_bar=False,
    )

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(train_emb, train_labels)

    val_preds = clf.predict(val_emb).tolist()
    metrics   = compute_metrics(val_preds, val_labels)
    elapsed   = time.time() - t0

    encoder.save(f"{save_dir}/encoder")
    with open(f"{save_dir}/classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    print_metrics(f"FrozenLR n={size} — val", metrics, elapsed)
    return {
        "approach": "frozen_lr", "size": size,
        "val_metrics": {k: v for k, v in metrics.items() if k != "report"},
        "train_time_s": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════════
# Approach 2 — SetFit
# ══════════════════════════════════════════════════════════════════════════
def setfit_iterations_for_size(size: int) -> int:
    schedule = {
        8: 30,
        64: 24,
        128: 20,
        250: 16,
        500: 12,
        1000: 8,
        2500: 5,
        5000: 4,
        7500: 3,
        12500: 2,
        15036: 2,
    }
    return schedule.get(size, max(1, min(30, round(30000 / size))))


def train_setfit(
    train_texts:    list[str],
    train_labels:   list[int],
    val_texts:      list[str],
    val_labels:     list[int],
    size:           int,
    num_epochs:     int = 4,
    num_iterations: Optional[int] = None,
) -> dict:
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    save_dir = _save_dir("setfit", size)
    t0       = time.time()

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds   = Dataset.from_dict({"text": val_texts,   "label": val_labels})
    if num_iterations is None:
        num_iterations = setfit_iterations_for_size(size)

    model = SetFitModel.from_pretrained(
        ENCODER_MODEL,
        processor_kwargs=TOKENIZER_KWARGS,
    )

    args = TrainingArguments(
        output_dir=f"{save_dir}/checkpoints",
        batch_size=16,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        metric="f1",
        column_mapping={"text": "text", "label": "label"},
    )

    print(
        f"  [SetFit] Training n={size} "
        f"({num_epochs} epochs, num_iterations={num_iterations}) …"
    )
    trainer.train()
    elapsed = time.time() - t0

    val_preds = model.predict(val_texts).tolist()
    metrics   = compute_metrics(val_preds, val_labels)

    model.save_pretrained(save_dir)
    print_metrics(f"SetFit n={size} — val (best ckpt)", metrics, elapsed)
    return {
        "approach": "setfit", "size": size,
        "val_metrics": {k: v for k, v in metrics.items() if k != "report"},
        "num_iterations": num_iterations,
        "train_time_s": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════════
# Approach 3 — Full encoder fine-tuning
# ══════════════════════════════════════════════════════════════════════════
def train_finetune(
    train_texts:  list[str],
    train_labels: list[int],
    val_texts:    list[str],
    val_labels:   list[int],
    size:         int,
    num_epochs:   int = 5,
    batch_size:   int = 32,
    lr:           float = 2e-5,
) -> dict:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from datasets import Dataset
    from sklearn.metrics import f1_score

    save_dir  = _save_dir("finetune", size)
    t0        = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_MODEL,
        **TOKENIZER_KWARGS,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True,
            padding="max_length", max_length=128,
        )

    train_ds = Dataset.from_dict(
        {"text": train_texts, "label": train_labels}
    ).map(tokenize, batched=True)
    val_ds   = Dataset.from_dict(
        {"text": val_texts, "label": val_labels}
    ).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODER_MODEL, num_labels=2,
    )

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "f1": float(f1_score(labels, preds, average="macro")),
        }

    training_args = TrainingArguments(
        output_dir=f"{save_dir}/checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"  [FineTune] Training n={size} (max {num_epochs} epochs, "
          f"patience=2) …")
    trainer.train()
    elapsed = time.time() - t0

    preds_out = trainer.predict(val_ds)
    val_preds = np.argmax(preds_out.predictions, axis=1).tolist()
    metrics   = compute_metrics(val_preds, val_labels)

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print_metrics(f"FineTune n={size} — val (best ckpt)", metrics, elapsed)
    return {
        "approach": "finetune", "size": size,
        "val_metrics": {k: v for k, v in metrics.items() if k != "report"},
        "train_time_s": round(elapsed, 1),
    }


# ── Learning curve runner ──────────────────────────────────────────────────
def _load_existing_results() -> list[dict]:
    if not Path(LEARNING_CURVE_PATH).exists():
        return []
    with open(LEARNING_CURVE_PATH, encoding="utf-8") as f:
        results = json.load(f)
    current_sizes = set(LEARNING_CURVE_SIZES)
    return [r for r in results if r.get("size") in current_sizes]


def _save_result(results: list[dict], result: dict) -> list[dict]:
    updated = [
        r for r in results
        if not (
            r["approach"] == result["approach"]
            and r["size"] == result["size"]
        )
    ]
    updated.append(result)
    updated.sort(key=lambda r: (r["approach"], r["size"]))
    with open(LEARNING_CURVE_PATH, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
    return updated


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _print_global_progress(
    done: int,
    total: int,
    started_at: float,
    timed_done: int,
) -> None:
    elapsed = time.time() - started_at
    fraction = done / total if total else 1.0
    filled = int(30 * fraction)
    bar = "#" * filled + "-" * (30 - filled)
    eta = (
        _format_duration((elapsed / timed_done) * (total - done))
        if timed_done > 0
        else "estimating"
    )
    print(
        f"[Router] Overall progress [{bar}] "
        f"{done}/{total} ({fraction * 100:.1f}%) | "
        f"elapsed={_format_duration(elapsed)} | "
        f"ETA={eta}",
        flush=True,
    )


def run_learning_curve(
    approaches:  list[str],
    sizes:       list[int],
    val_texts:   list[str],
    val_labels:  list[int],
    seed:        int = 42,
) -> list[dict]:
    """Run all approach × size combinations and return results."""

    # Load full training data once
    all_train_texts, all_train_labels = load_split(
        ROUTER_TRAIN_SQAC, ROUTER_TRAIN_CHITCHAT,
    )
    print(f"\n[Router] Full training set: {len(all_train_texts):,} samples")
    print(f"[Router] Val set          : {len(val_texts):,} samples")
    print(f"[Router] Approaches       : {approaches}")
    print(f"[Router] Sizes (per class): {sizes}")
    planned_runs = [
        (approach, size)
        for approach in approaches
        for size in sizes
    ]
    total_runs = len(planned_runs)
    print(f"[Router] Total runs       : {total_runs}\n")

    all_results = _load_existing_results()
    completed_keys = set()
    for r in all_results:
        key = (r["approach"], r["size"])
        if r["approach"] == "setfit":
            expected_iterations = setfit_iterations_for_size(r["size"])
            if r.get("num_iterations") != expected_iterations:
                continue
        completed_keys.add(key)
    run_started_at = time.time()
    completed_runs = sum(
        1 for approach, size in planned_runs
        if (approach, size) in completed_keys
    )
    initially_completed_runs = completed_runs

    if completed_runs:
        print(
            f"[Router] Found {completed_runs}/{total_runs} completed runs "
            "in learning_curve.json; skipping them.",
            flush=True,
        )
        _print_global_progress(
            completed_runs,
            total_runs,
            run_started_at,
            timed_done=0,
        )

    for approach in approaches:
        print(f"\n{'═' * 60}")
        print(f"  Approach: {approach}")
        print(f"{'═' * 60}")

        for size in sizes:
            if (approach, size) in completed_keys:
                print(
                    f"\n  ── Skipping completed run: {approach} "
                    f"size={size} ──",
                    flush=True,
                )
                continue

            print(
                f"\n  ── Run {completed_runs + 1}/{total_runs}: "
                f"{approach} size={size} per class ({size * 2} total) ──",
                flush=True,
            )

            # Subsample training data for this size
            train_texts, train_labels = load_split(
                ROUTER_TRAIN_SQAC, ROUTER_TRAIN_CHITCHAT,
                n_per_class=size, seed=seed,
            )

            if approach == "frozen_lr":
                res = train_frozen_lr(
                    train_texts, train_labels, val_texts, val_labels, size,
                )
            elif approach == "setfit":
                res = train_setfit(
                    train_texts, train_labels, val_texts, val_labels, size,
                )
            elif approach == "finetune":
                res = train_finetune(
                    train_texts, train_labels, val_texts, val_labels, size,
                )

            # Save incrementally so partial results are not lost
            all_results = _save_result(all_results, res)
            completed_keys.add((approach, size))
            completed_runs += 1
            _print_global_progress(
                completed_runs,
                total_runs,
                run_started_at,
                timed_done=completed_runs - initially_completed_runs,
            )

    return all_results


# ── Learning curve summary table ───────────────────────────────────────────
def print_learning_curve_table(results: list[dict]) -> None:
    approaches = sorted(set(r["approach"] for r in results))
    sizes      = sorted(set(r["size"] for r in results))

    # Index results
    idx = {(r["approach"], r["size"]): r for r in results}

    col_w = 10
    head_w = 12

    for approach in approaches:
        print(f"\n── {approach} ─────────────────────────────────────────────")
        header = f"{'Size':<{head_w}}" + f"{'MacroF1':>{col_w}}" + \
                 f"{'F1_no_ret':>{col_w}}" + \
                 f"{'F1_ret':>{col_w}}" + f"{'Time(s)':>{col_w}}"
        print(header)
        print("─" * (head_w + col_w * 4))
        for size in sizes:
            r = idx.get((approach, size))
            if r is None:
                continue
            m    = r["val_metrics"]
            f1pc = m.get("f1_per_class", [None, None])
            print(
                f"{size:<{head_w}}"
                f"{m['macro_f1']:>{col_w}.4f}"
                f"{f1pc[0]:>{col_w}.4f}"
                f"{f1pc[1]:>{col_w}.4f}"
                f"{r['train_time_s']:>{col_w}.1f}"
            )

    print(f"\n[Router] Best configurations by val macro F1:")
    print("─" * 50)
    for approach in approaches:
        best = max(
            (r for r in results if r["approach"] == approach),
            key=lambda r: r["val_metrics"]["macro_f1"],
        )
        print(
            f"  {approach:<12}  size={best['size']:<6}  "
            f"F1={best['val_metrics']['macro_f1']:.4f}"
        )
    print("─" * 50)
    print("  → Run router_oracle.py with the best configuration.")


# ══════════════════════════════════════════════════════════════════════════
# QueryRouter — inference wrapper used by main.py
# ══════════════════════════════════════════════════════════════════════════
class QueryRouter:
    """
    Loads the best saved router model and exposes needs_retrieval(query).

    Automatically selects the best approach + size from learning_curve.json
    if available. Falls back through finetune → setfit → frozen_lr →
    rule-based.

    Override with approach= and size= to force a specific model.
    """

    def __init__(
        self,
        approach: Optional[str] = None,
        size:     Optional[int] = None,
        model_dir: Optional[str] = None,
    ):
        self._predict_fn: Optional[Callable] = None
        self._approach:   Optional[str]      = None

        if model_dir:
            if self._load_from_dir(approach or "setfit", model_dir):
                self._approach = approach or "setfit"
                print(f"[Router] Loaded: {self._approach} ({model_dir})")
            else:
                print(f"[Router] Could not load model_dir={model_dir}")
            if self._predict_fn is None:
                print("[Router] No trained model found — using rule-based fallback.")
            return

        # Auto-select best from learning curve results if not specified
        if approach is None or size is None:
            approach, size = self._best_from_curve(approach, size)

        candidates = (
            [(approach, size)]
            if approach is not None and size is not None
            else [("finetune", None), ("setfit", None), ("frozen_lr", None)]
        )

        for cand_approach, cand_size in candidates:
            if self._try_load(cand_approach, cand_size):
                self._approach = cand_approach
                print(f"[Router] Loaded: {cand_approach} (size={cand_size})")
                break

        if self._predict_fn is None:
            print("[Router] No trained model found — using rule-based fallback.")

    def _best_from_curve(
        self,
        approach: Optional[str],
        size:     Optional[int],
    ) -> tuple[Optional[str], Optional[int]]:
        """Read learning_curve.json and return best approach + size."""
        if not Path(LEARNING_CURVE_PATH).exists():
            return approach, size
        with open(LEARNING_CURVE_PATH, encoding="utf-8") as f:
            results = json.load(f)
        if not results:
            return approach, size
        # Filter by approach if specified
        candidates = [
            r for r in results
            if (approach is None or r["approach"] == approach)
            and (size is None or r["size"] == size)
        ]
        if not candidates:
            return approach, size
        best = max(candidates, key=lambda r: r["val_metrics"]["macro_f1"])
        return best["approach"], best["size"]

    def _try_load(self, approach: str, size: Optional[int]) -> bool:
        """Try to load a saved model for the given approach and size."""
        # Find the right directory
        if size is not None:
            dirs = [f"{MODEL_DIR}/{approach}/{size}"]
        else:
            # Try all sizes, pick best by learning curve
            base = Path(f"{MODEL_DIR}/{approach}")
            if not base.exists():
                return False
            dirs = sorted(
                [str(d) for d in base.iterdir() if d.is_dir()],
                key=lambda d: int(Path(d).name) if Path(d).name.isdigit() else 0,
                reverse=True,
            )

        for save_dir in dirs:
            if self._load_from_dir(approach, save_dir):
                return True
        return False

    def _load_from_dir(self, approach: str, save_dir: str) -> bool:
        try:
            if approach == "frozen_lr":
                from sentence_transformers import SentenceTransformer
                enc_path = Path(f"{save_dir}/encoder")
                clf_path = Path(f"{save_dir}/classifier.pkl")
                if not enc_path.exists() or not clf_path.exists():
                    return False
                encoder = SentenceTransformer(
                    str(enc_path),
                    processor_kwargs=TOKENIZER_KWARGS,
                )
                with open(clf_path, "rb") as f:
                    clf = pickle.load(f)
                def _predict(queries: list[str]) -> list[int]:
                    emb = encoder.encode(queries, normalize_embeddings=True)
                    return clf.predict(emb).tolist()
                self._predict_fn = _predict
                return True

            elif approach == "setfit":
                if not Path(save_dir).exists():
                    return False
                from setfit import SetFitModel
                model = SetFitModel.from_pretrained(
                    save_dir,
                    processor_kwargs=TOKENIZER_KWARGS,
                )
                self._predict_fn = lambda qs: model.predict(qs).tolist()
                return True

            elif approach == "finetune":
                if not Path(save_dir).exists():
                    return False
                import torch
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSequenceClassification,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    save_dir,
                    **TOKENIZER_KWARGS,
                )
                model     = AutoModelForSequenceClassification.from_pretrained(
                    save_dir,
                )
                model.eval()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)

                def _predict(queries: list[str]) -> list[int]:
                    enc = tokenizer(
                        queries, truncation=True, padding=True,
                        max_length=128, return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        logits = model(**enc).logits
                    return torch.argmax(logits, dim=1).cpu().tolist()

                self._predict_fn = _predict
                return True

        except Exception as exc:
            print(f"[Router] Could not load {approach} from {save_dir}: {exc}")
        return False

    def needs_retrieval(self, query: str) -> bool:
        if self._predict_fn is not None:
            return int(self._predict_fn([query])[0]) == 1
        return rule_based_router(query) == 1

    def predict_batch(self, queries: list[str]) -> list[int]:
        if self._predict_fn is not None:
            return [int(p) for p in self._predict_fn(queries)]
        return [rule_based_router(q) for q in queries]


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Query Router — learning curve across data sizes."
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Run learning curve training.",
    )
    parser.add_argument(
        "--approach",
        choices=["frozen_lr", "setfit", "finetune", "all"],
        default="all",
        help="Which approach(es) to train (default: all).",
    )
    parser.add_argument(
        "--size", type=int, default=None,
        help="Train a single data size (per class). Default: all sizes.",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Classify a single query using the best saved model.",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run rule-based router on hardcoded examples.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke_test:
        examples = [
            ("¿Quién fue Cervantes?",             1),
            ("Hola, ¿cómo estás?",                0),
            ("¿Cuándo murió Frida Kahlo?",         1),
            ("Buenos días, ¿me ayudas con algo?",  0),
            ("¿Qué es la fotosíntesis?",           1),
            ("Escribe un poema sobre el mar.",     0),
        ]
        print("\n── Rule-based smoke test ──────────────────────────────")
        for query, gold in examples:
            pred = rule_based_router(query)
            mark = "✓" if pred == gold else "✗"
            print(f"  {mark}  [gold={gold} pred={pred}]  {query}")
        return

    if args.train:
        val_texts, val_labels = load_split(
            ROUTER_VAL_SQAC, ROUTER_VAL_CHITCHAT,
        )

        approaches = (
            ["frozen_lr", "setfit", "finetune"]
            if args.approach == "all"
            else [args.approach]
        )

        sizes = (
            [args.size] if args.size is not None
            else LEARNING_CURVE_SIZES
        )

        results = run_learning_curve(approaches, sizes, val_texts, val_labels)
        print_learning_curve_table(results)

        print(f"\n[Router] Learning curve saved → {LEARNING_CURVE_PATH}")
        print("[Router] Next: inspect results, then run router_oracle.py "
              "with the best approach + size.")

    if args.query:
        router   = QueryRouter()
        decision = router.needs_retrieval(args.query)
        print(f"\nQuery   : {args.query}")
        print(f"Decision: {'RETRIEVE' if decision else 'NO RETRIEVE'}")
        print(f"Model   : {router._approach or 'rule-based'}")


if __name__ == "__main__":
    main()
