"""
evaluate.py
-----------
Evaluation suite for the RAG Conversational System.

Reads prediction JSONL files written by main.py (one per strategy) and
computes the following metrics:

Response Quality
────────────────
  Exact Match (EM)       — strict string match after normalisation
  Token F1               — token-level overlap (SQuAD-style)
  BERTScore F1           — semantic similarity via multilingual BERT

Retrieval Quality  (only for samples where retrieval was triggered)
──────────────────
  Recall@k               — gold doc in top-k retrieved passages
  Precision@k            — fraction of retrieved passages that are relevant
  MRR                    — Mean Reciprocal Rank of the gold document

Router Performance  (router_rag strategy only)
──────────────────
  Decision Accuracy      — correct retrieve / skip decisions
  False Retrieval Rate   — skip when retrieval was needed
  Missed Retrieval Rate  — retrieve when retrieval was not needed

Latency
───────
  Mean total latency per sample (seconds)

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
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = "./rag_project"
DATA_DIR   = f"{BASE_DIR}/data"
OUTPUT_DIR = f"{BASE_DIR}/outputs"
EVAL_PATH  = f"{DATA_DIR}/sqac_validation.jsonl"
PRED_TPL   = f"{OUTPUT_DIR}/predictions_{{strategy}}.jsonl"
REPORT_DIR = f"{OUTPUT_DIR}/reports"
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

STRATEGIES = ["always_rag", "never_rag", "router_rag"]

# ── Text normalisation (SQuAD-style) ──────────────────────────────────────
def _normalize(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ── EM & Token F1 ─────────────────────────────────────────────────────────
def exact_match(prediction: str, gold_list: list[str]) -> float:
    pred_norm = _normalize(prediction)
    return float(any(_normalize(g) == pred_norm for g in gold_list))


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


# ── BERTScore ──────────────────────────────────────────────────────────────
def compute_bertscore(
    predictions: list[str],
    references:  list[str],
    lang:        str = "es",
    batch_size:  int = 16,
) -> list[float]:
    """
    Compute BERTScore F1 using bert_score library.
    Falls back to token F1 if bert_score is not installed.

    Parameters
    ----------
    predictions : model outputs (one per sample)
    references  : single best gold answer per sample
    lang        : ISO language code for BERTScore model selection
    batch_size  : BERTScore internal batch size

    Returns
    -------
    List of per-sample F1 scores in [0, 1].
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
        return [
            token_f1(p, [r]) for p, r in zip(predictions, references)
        ]


# ── Retrieval metrics ──────────────────────────────────────────────────────
def recall_at_k(retrieved_ids: list[str], gold_id: str) -> float:
    """1.0 if gold_id appears in retrieved_ids, else 0.0."""
    return float(gold_id in retrieved_ids)


def precision_at_k(retrieved_ids: list[str], gold_id: str) -> float:
    """Fraction of retrieved passages that match the gold document."""
    if not retrieved_ids:
        return 0.0
    return sum(1 for rid in retrieved_ids if rid == gold_id) / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: list[str], gold_id: str) -> float:
    """Reciprocal rank of the gold document in the ranked list."""
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid == gold_id:
            return 1.0 / rank
    return 0.0


# ── Router performance ─────────────────────────────────────────────────────
def router_metrics(
    records:          list[dict],
    gold_labels:      dict[str, int],   # sample_id → 1 (retrieve) / 0 (skip)
) -> dict:
    """
    Compute router-specific metrics.

    Parameters
    ----------
    records     : prediction records (must have 'sample_id' and 'do_retrieve').
    gold_labels : ground-truth retrieval necessity per sample.

    Returns
    -------
    dict with accuracy, false_retrieval_rate, missed_retrieval_rate.
    """
    n = 0
    correct = false_ret = missed_ret = 0

    for rec in records:
        sid  = rec.get("sample_id", "")
        if sid not in gold_labels:
            continue
        gold = gold_labels[sid]
        pred = int(rec.get("do_retrieve", True))
        n += 1
        if pred == gold:
            correct += 1
        elif pred == 1 and gold == 0:
            false_ret += 1      # retrieved when not needed
        elif pred == 0 and gold == 1:
            missed_ret += 1     # skipped when retrieval was needed

    if n == 0:
        return {"accuracy": None, "false_retrieval_rate": None,
                "missed_retrieval_rate": None, "n_labelled": 0}
    return {
        "accuracy":             correct   / n,
        "false_retrieval_rate": false_ret / n,
        "missed_retrieval_rate":missed_ret/ n,
        "n_labelled":           n,
    }


# ── Load predictions ───────────────────────────────────────────────────────
def load_predictions(strategy: str) -> list[dict]:
    path = PRED_TPL.format(strategy=strategy)
    if not Path(path).exists():
        print(f"[Evaluate] Predictions file not found: {path} — skipping.")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_gold_labels(eval_path: str = EVAL_PATH) -> dict[str, int]:
    """
    Build ground-truth retrieval labels from the evaluation set.
    All SQAC questions are factual → label = 1 (retrieval needed).
    This can be replaced with manually annotated labels if available.
    """
    if not Path(eval_path).exists():
        return {}
    with open(eval_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]
    # All SQAC questions are knowledge-intensive: gold label = 1
    return {s["id"]: 1 for s in samples}


# ── Per-strategy evaluation ────────────────────────────────────────────────
def evaluate_strategy(
    strategy:       str,
    gold_labels:    dict[str, int],
    verbose:        bool = False,
) -> dict:
    records = load_predictions(strategy)
    if not records:
        return {}

    em_scores, f1_scores, latencies = [], [], []
    recall_scores, precision_scores, rr_scores = [], [], []
    preds_for_bert, refs_for_bert = [], []

    for rec in records:
        gold_answers = rec.get("gold_answers", [])
        answer       = rec.get("answer", "")
        gold_doc     = rec.get("gold_doc_id", "")
        retrieved    = rec.get("passages_retrieved", [])

        if not gold_answers:
            continue

        # Response quality
        em = exact_match(answer, gold_answers)
        f1 = token_f1(answer, gold_answers)
        em_scores.append(em)
        f1_scores.append(f1)

        # BERTScore inputs (use first gold answer as reference)
        preds_for_bert.append(answer)
        refs_for_bert.append(gold_answers[0])

        # Retrieval metrics (only when retrieval was actually triggered)
        if rec.get("effective_retrieved", False) and retrieved:
            recall_scores.append(recall_at_k(retrieved, gold_doc))
            precision_scores.append(precision_at_k(retrieved, gold_doc))
            rr_scores.append(reciprocal_rank(retrieved, gold_doc))

        latencies.append(rec.get("total_latency_s", 0.0))

        if verbose:
            print(
                f"  [{rec.get('sample_id','')}] EM={em:.0f} F1={f1:.2f} "
                f"dec={rec.get('router_decision','')} lat={latencies[-1]:.2f}s"
            )

    # BERTScore
    bert_scores = compute_bertscore(preds_for_bert, refs_for_bert) if preds_for_bert else []

    # Router metrics (only meaningful for router_rag)
    router_met = {}
    if strategy == "router_rag":
        router_met = router_metrics(records, gold_labels)

    n = len(em_scores)
    result = {
        "strategy":               strategy,
        "n_samples":              n,
        "exact_match":            float(np.mean(em_scores))            if em_scores    else None,
        "token_f1":               float(np.mean(f1_scores))            if f1_scores    else None,
        "bertscore_f1":           float(np.mean(bert_scores))          if bert_scores  else None,
        "recall_at_k":            float(np.mean(recall_scores))        if recall_scores   else None,
        "precision_at_k":         float(np.mean(precision_scores))     if precision_scores else None,
        "mrr":                    float(np.mean(rr_scores))            if rr_scores    else None,
        "mean_latency_s":         float(np.mean(latencies))            if latencies    else None,
        "pct_retrieved":          float(np.mean([
                                      int(r.get("effective_retrieved", False))
                                      for r in records
                                  ])) if records else None,
        **router_met,
    }
    return result


# ── Summary table ──────────────────────────────────────────────────────────
def _fmt(val) -> str:
    if val is None:
        return "  N/A  "
    return f"{val:6.4f}"


def print_summary_table(results: list[dict]) -> None:
    col_w = 26
    metrics = [
        ("Exact Match",            "exact_match"),
        ("Token F1",               "token_f1"),
        ("BERTScore F1",           "bertscore_f1"),
        ("Recall@k",               "recall_at_k"),
        ("Precision@k",            "precision_at_k"),
        ("MRR",                    "mrr"),
        ("Mean Latency (s)",       "mean_latency_s"),
        ("% Samples Retrieved",    "pct_retrieved"),
        ("Router Accuracy",        "accuracy"),
        ("False Retrieval Rate",   "false_retrieval_rate"),
        ("Missed Retrieval Rate",  "missed_retrieval_rate"),
    ]

    strats = [r["strategy"] for r in results]
    header = f"{'Metric':<{col_w}}" + "".join(f"{s:>12}" for s in strats)
    sep    = "─" * len(header)

    print("\n" + sep)
    print("  Evaluation Results")
    print(sep)
    print(header)
    print(sep)
    for label, key in metrics:
        row = f"{label:<{col_w}}"
        for r in results:
            row += f"{_fmt(r.get(key)):>12}"
        print(row)
    print(sep + "\n")


# ── Save report ────────────────────────────────────────────────────────────
def save_report(results: list[dict]) -> None:
    report_path = f"{REPORT_DIR}/evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Evaluate] Full report saved → {report_path}")


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
    args        = _parse_args()
    gold_labels = load_gold_labels()

    strategies_to_eval = (
        STRATEGIES if args.strategy == "all" else [args.strategy]
    )

    all_results = []
    for strat in strategies_to_eval:
        print(f"\n[Evaluate] Strategy: {strat}")
        res = evaluate_strategy(strat, gold_labels, verbose=args.verbose)
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
