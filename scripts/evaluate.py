"""
evaluate.py
-----------
Evaluation suite for the RAG Conversational System.

Reads prediction JSONL files written by main.py (one per strategy) and
computes the following metrics:

Response Quality
────────────────
  Token F1               — token-level overlap (SQuAD-style)
  BERTScore F1           — semantic similarity via multilingual BERT

Retrieval Quality  (only for samples where retrieval was triggered)
──────────────────
  Recall@3               — is the gold passage in the top-3 retrieved?
  MRR                    — Mean Reciprocal Rank of the gold passage

Router Performance  (router_rag strategy only)
──────────────────
  Retrieval Rate         — % of queries where router chose RETRIEVE
  Oracle Accuracy        — router correct vs. optimal route (by token F1)

Usage
─────
  # Evaluate all three strategies
  python evaluate.py

  # Evaluate a single strategy
  python evaluate.py --strategy router_rag

  # Print a detailed per-sample breakdown
  python evaluate.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import numpy as np
from config import HF_TOKEN

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = "./data/processed"
OUTPUT_DIR = "./outputs"
EVAL_PATH  = f"{DATA_DIR}/sqac_validation.jsonl"
PRED_TPL   = f"{OUTPUT_DIR}/predictions_{{strategy}}.jsonl"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

STRATEGIES = ["always_rag", "never_rag", "router_rag"]

# ── Text normalisation (SQuAD-style) ──────────────────────────────────────
def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ── Token F1 ──────────────────────────────────────────────────────────────
def token_f1(prediction: str, gold_list: list[str]) -> float:
    """SQuAD-style token F1: best F1 over all gold answers."""
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in gold_list:
        gold_tokens = _tokenize(gold)
        if not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        prec = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common) / len(pred_tokens)
        rec  = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common) / len(gold_tokens)
        if prec + rec == 0:
            continue
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return best


# ── BERTScore ─────────────────────────────────────────────────────────────
def compute_bertscore(
    predictions: list[str],
    references:  list[str],
    lang:        str = "es",
    batch_size:  int = 16,
) -> list[float]:
    """
    Compute BERTScore F1 using bert_score library.
    Falls back to token F1 if bert_score is not installed.
    """
    try:
        from bert_score import score as bs_score
        _, _, F = bs_score(
            predictions,
            references,
            lang=lang,
            batch_size=batch_size,
            verbose=False,
        )
        return F.tolist()
    except ImportError:
        print("[Evaluate] bert_score not installed — using token F1 as proxy.")
        return [token_f1(p, [r]) for p, r in zip(predictions, references)]


# ── Retrieval metrics ──────────────────────────────────────────────────────
def recall_at_3(retrieved_ids: list[str], gold_id: str) -> float:
    """1.0 if gold_id appears in the top-3 retrieved passages, else 0.0."""
    return float(gold_id in retrieved_ids[:3])


def reciprocal_rank(retrieved_ids: list[str], gold_id: str) -> float:
    """Reciprocal rank of the gold document in the ranked list."""
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid == gold_id:
            return 1.0 / rank
    return 0.0


# ── Router metrics ─────────────────────────────────────────────────────────
def oracle_accuracy(
    records:     list[dict],
    f1_scores:   list[float],
    threshold:   float = 0.0,
) -> float:
    """
    Oracle accuracy: fraction of queries where the router's decision matches
    the optimal route determined by token F1.

    The optimal route is defined as:
      - RETRIEVE (1) if always_rag token F1 > never_rag token F1 for that sample
      - SKIP     (0) otherwise

    Since we don't have per-sample cross-strategy F1 here, we approximate:
      - optimal = RETRIEVE (1) if the sample's token F1 under the current
        strategy is above the threshold, meaning retrieval helped.
      - For never_rag records, optimal is always SKIP (0).

    A simpler and standard approximation used in RAG literature:
      optimal = 1 (retrieve) if token F1 > threshold, else 0
      correct  = router decision matches optimal
    """
    if not records or not f1_scores:
        return None

    correct = 0
    for rec, f1 in zip(records, f1_scores):
        optimal   = 1 if f1 > threshold else 0
        predicted = int(rec.get("do_retrieve", True))
        if predicted == optimal:
            correct += 1
    return correct / len(records)


# ── Load helpers ───────────────────────────────────────────────────────────
def load_predictions(strategy: str) -> list[dict]:
    path = PRED_TPL.format(strategy=strategy)
    if not Path(path).exists():
        print(f"[Evaluate] Predictions file not found: {path} — skipping.")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ── Per-strategy evaluation ────────────────────────────────────────────────
def evaluate_strategy(
    strategy: str,
    verbose:  bool = False,
) -> dict:
    records = load_predictions(strategy)
    if not records:
        return {}

    f1_scores, rr_scores, recall_scores = [], [], []
    preds_for_bert, refs_for_bert = [], []
    n_retrieved = 0

    for rec in records:
        gold_answers = rec.get("gold_answers", [])
        answer       = rec.get("answer", "")
        gold_doc     = rec.get("gold_doc_id", "")
        retrieved    = rec.get("passages_retrieved", [])

        if not gold_answers:
            continue

        # ── Token F1 ──────────────────────────────────────────────────────
        f1 = token_f1(answer, gold_answers)
        f1_scores.append(f1)

        # ── BERTScore inputs ───────────────────────────────────────────────
        preds_for_bert.append(answer)
        refs_for_bert.append(gold_answers[0])

        # ── Retrieval metrics (only when retrieval was triggered) ──────────
        if rec.get("effective_retrieved", False) and retrieved:
            recall_scores.append(recall_at_3(retrieved, gold_doc))
            rr_scores.append(reciprocal_rank(retrieved, gold_doc))

        # ── Retrieval rate counter ─────────────────────────────────────────
        if rec.get("do_retrieve", False):
            n_retrieved += 1

        if verbose:
            print(
                f"  [{rec.get('sample_id', '')}] "
                f"F1={f1:.2f}  dec={rec.get('router_decision', '')}  "
                f"retrieved={rec.get('effective_retrieved', False)}"
            )

    # ── BERTScore ──────────────────────────────────────────────────────────
    bert_scores = compute_bertscore(preds_for_bert, refs_for_bert) if preds_for_bert else []

    # ── Retrieval rate ─────────────────────────────────────────────────────
    retrieval_rate = n_retrieved / len(records) if records else None

    # ── Oracle accuracy (router_rag only) ─────────────────────────────────
    oracle_acc = None
    if strategy == "router_rag":
        oracle_acc = oracle_accuracy(records, f1_scores)

    result = {
        "strategy":        strategy,
        "n_samples":       len(f1_scores),
        "token_f1":        float(np.mean(f1_scores))     if f1_scores      else None,
        "bertscore_f1":    float(np.mean(bert_scores))   if bert_scores    else None,
        "recall_at_3":     float(np.mean(recall_scores)) if recall_scores  else None,
        "mrr":             float(np.mean(rr_scores))     if rr_scores      else None,
        "retrieval_rate":  retrieval_rate,
        "oracle_accuracy": oracle_acc,
    }
    return result


# ── Summary table ──────────────────────────────────────────────────────────
def _fmt(val) -> str:
    if val is None:
        return "  N/A  "
    return f"{val:6.4f}"


def print_summary_table(results: list[dict]) -> None:
    col_w = 22
    metrics = [
        ("Token F1",          "token_f1"),
        ("BERTScore F1",      "bertscore_f1"),
        ("Recall@3",          "recall_at_3"),
        ("MRR",               "mrr"),
        ("Retrieval Rate",    "retrieval_rate"),
        ("Oracle Accuracy",   "oracle_accuracy"),
    ]

    strats = [r["strategy"] for r in results]
    header = f"{'Metric':<{col_w}}" + "".join(f"{s:>16}" for s in strats)
    sep    = "─" * len(header)

    print("\n" + sep)
    print("  Evaluation Results")
    print(sep)
    print(header)
    print(sep)
    for label, key in metrics:
        row = f"{label:<{col_w}}"
        for r in results:
            row += f"{_fmt(r.get(key)):>16}"
        print(row)
    print(sep + "\n")


# ── Save report ────────────────────────────────────────────────────────────
def save_report(results: list[dict]) -> None:
    report_path = f"{OUTPUT_DIR}/evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Evaluate] Report saved → {report_path}")


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline predictions."
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES + ["all"],
        default="all",
        help="Which strategy to evaluate (default: all).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample breakdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    strategies_to_eval = (
        STRATEGIES if args.strategy == "all" else [args.strategy]
    )

    all_results = []
    for strat in strategies_to_eval:
        print(f"\n[Evaluate] Strategy: {strat}")
        res = evaluate_strategy(strat, verbose=args.verbose)
        if res:
            all_results.append(res)

    if all_results:
        print_summary_table(all_results)
        save_report(all_results)
    else:
        print(
            "\n[Evaluate] No predictions found. "
            "Run main.py --mode batch --strategy <name> first."
        )


if __name__ == "__main__":
    main()