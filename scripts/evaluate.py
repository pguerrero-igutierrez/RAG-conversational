"""
evaluate.py
-----------
Full evaluation suite for the RAG Conversational System.

Reads the unified test set (test_sqac.jsonl + test_chitchat.jsonl, 170
samples total by default) and computes all metrics across pipeline strategies.

Metrics
-------
  Pre-feedback retrieval decision performance  (all test samples)
  ──────────────────────────────
    Accuracy               — correct initial retrieve/skip decisions
    Macro F1               — balanced F1 across both classes
    F1 per class           — F1 for no_retrieval and retrieval separately

  Post-feedback effective retrieval performance  (all test samples)
  ──────────────────────────────
    Accuracy / Macro F1    — final effective retrieve/skip state after
                             self-feedback can discard or re-retrieve context

  Oracle policy evaluation  (SQAC samples only)
  ────────────────────────
    Oracle accuracy        — retrieval decisions vs oracle labels derived by
                             comparing always_rag vs never_rag BERTScore F1
                             per sample. Oracle label=1 if always_rag
                             BERTScore F1 > never_rag BERTScore F1, else 0.

  Response Quality  (SQAC samples only)
  ────────────────
    Token F1               — SQuAD-style lexical overlap with gold answers
    BERTScore F1           — semantic similarity via multilingual BERT

  Retrieval Quality  (SQAC samples where retrieval was triggered)
  ─────────────────
    Recall@3               — gold passage in top-3 retrieved
    MRR                    — Mean Reciprocal Rank of gold passage

  Latency
  ───────
    Mean total latency per sample (seconds)
    % samples where retrieval was triggered

Workflow
--------
  1. Run batch predictions for all three strategies:
       python scripts/main.py --mode batch --strategy always_rag
       python scripts/main.py --mode batch --strategy never_rag
       python scripts/main.py --mode batch --strategy router_rag

  2. Run evaluation:
       python scripts/evaluate.py
       python scripts/evaluate.py --strategy router_rag
       python scripts/evaluate.py --verbose

Output
------
  outputs/evaluation_report.json   — full results JSON
  Printed summary table to stdout
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = "./data/processed"
OUTPUT_DIR    = "./outputs"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

TEST_SQAC_PATH     = f"{PROCESSED_DIR}/test_sqac.jsonl"
TEST_CHITCHAT_PATH = f"{PROCESSED_DIR}/test_chitchat.jsonl"
PRED_TPL           = f"{OUTPUT_DIR}/predictions_{{strategy}}.jsonl"
REPORT_PATH        = f"{OUTPUT_DIR}/evaluation_report.json"

BASELINE_STRATEGIES = ["always_rag", "never_rag"]


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
        prec = sum(
            min(pred_tokens.count(t), gold_tokens.count(t)) for t in common
        ) / len(pred_tokens)
        rec = sum(
            min(pred_tokens.count(t), gold_tokens.count(t)) for t in common
        ) / len(gold_tokens)
        if prec + rec == 0:
            continue
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# ── BERTScore ─────────────────────────────────────────────────────────────
def compute_bertscore(
    predictions: list[str],
    references:  list[str],
    lang:        str = "es",
    batch_size:  int = 16,
) -> list[float]:
    try:
        from bert_score import score as bs_score
        _, _, F = bs_score(
            predictions, references,
            lang=lang, batch_size=batch_size, verbose=False,
        )
        return F.tolist()
    except ImportError:
        print("[Evaluate] bert_score not installed — using token F1 as proxy.")
        return [token_f1(p, [r]) for p, r in zip(predictions, references)]


# ── Retrieval metrics ──────────────────────────────────────────────────────
def recall_at_3(retrieved_ids: list[str], gold_id: str) -> float:
    return float(gold_id in retrieved_ids[:3])


def reciprocal_rank(retrieved_ids: list[str], gold_id: str) -> float:
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid == gold_id:
            return 1.0 / rank
    return 0.0


# ── Retrieval decision metrics ─────────────────────────────────────────────
def retrieval_decision_classification_metrics(
    records:     list[dict],
    gold_labels: dict[str, int],
    decision_key: str = "do_retrieve",
) -> dict:
    """
    Compute accuracy and F1 for retrieval decisions on the unified test set.
    gold_labels: sample_id → 0 or 1 (from test set ground truth).
    decision_key: record field containing the binary retrieve/skip decision.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    preds, golds = [], []
    for rec in records:
        sid = rec.get("sample_id", "")
        if sid not in gold_labels:
            continue
        golds.append(gold_labels[sid])
        preds.append(int(rec.get(decision_key, rec.get("do_retrieve", True))))

    if not preds:
        return {}

    f1pc   = f1_score(golds, preds, average=None, zero_division=0).tolist()
    report = classification_report(
        golds, preds,
        target_names=["no_retrieval", "retrieval"],
        zero_division=0,
    )
    return {
        "accuracy":     round(float(accuracy_score(golds, preds) * 100), 4),
        "macro_f1":     round(
            float(
                f1_score(golds, preds, average="macro", zero_division=0)
                * 100
            ),
            4,
        ),
        "f1_per_class": [round(float(x * 100), 4) for x in f1pc],
        "report":       report,
        "n_samples":    len(preds),
    }


# ── Oracle policy evaluation ───────────────────────────────────────────────
def compute_oracle_policy_accuracy(
    policy_records:      list[dict],
    always_rag_records:  list[dict],
    never_rag_records:   list[dict],
    decision_key: str = "do_retrieve",
    prefix: str = "oracle_policy",
) -> dict:
    """
    Oracle label per SQAC sample:
      oracle=1 (retrieve) if always_rag BERTScore F1 > never_rag BERTScore F1
      oracle=0 (skip)     otherwise

    Policy accuracy = fraction of samples where a model's retrieval decision
    matches the oracle label.
    """
    # Index by sample_id
    always_idx = {
        r["sample_id"]: r for r in always_rag_records
        if "gold_answers" in r and r["gold_answers"]
    }
    never_idx = {
        r["sample_id"]: r for r in never_rag_records
        if "gold_answers" in r and r["gold_answers"]
    }
    policy_idx = {r["sample_id"]: r for r in policy_records}

    common_ids = sorted(set(always_idx) & set(never_idx) & set(policy_idx))
    if not common_ids:
        print("[Evaluate] No common sample IDs for oracle policy evaluation.")
        return {}

    oracle_labels:  dict[str, int] = {}
    correct = 0

    references = [always_idx[sid]["gold_answers"][0] for sid in common_ids]
    always_predictions = [
        always_idx[sid].get("answer", "") for sid in common_ids
    ]
    never_predictions = [
        never_idx[sid].get("answer", "") for sid in common_ids
    ]

    always_scores = compute_bertscore(always_predictions, references)
    never_scores = compute_bertscore(never_predictions, references)

    for sid, always_score, never_score in zip(
        common_ids,
        always_scores,
        never_scores,
    ):
        policy_rec = policy_idx[sid]
        oracle = 1 if always_score > never_score else 0
        oracle_labels[sid] = oracle

        policy_decision = int(
            policy_rec.get(decision_key, policy_rec.get("do_retrieve", True))
        )
        if policy_decision == oracle:
            correct += 1

    n = len(common_ids)
    n_retrieve = sum(oracle_labels.values())
    return {
        f"{prefix}_accuracy": round((correct / n) * 100, 4) if n else None,
        f"{prefix}_n_samples": n,
        f"{prefix}_retrieve_rate": (
            round((n_retrieve / n) * 100, 4) if n else None
        ),
        f"{prefix}_score_metric": "bertscore_f1",
    }


# ── Load helpers ───────────────────────────────────────────────────────────
def load_predictions(strategy: str) -> list[dict]:
    path = PRED_TPL.format(strategy=strategy)
    if not Path(path).exists():
        print(f"[Evaluate] Predictions not found: {path} — skipping.")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_test_labels() -> dict[str, int]:
    """
    Build ground-truth retrieval-need labels from the unified test set.
    SQAC questions → label=1, chitchat prompts → label=0.
    """
    labels: dict[str, int] = {}

    if Path(TEST_SQAC_PATH).exists():
        with open(TEST_SQAC_PATH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                labels[r["id"]] = 1

    if Path(TEST_CHITCHAT_PATH).exists():
        with open(TEST_CHITCHAT_PATH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                labels[r["id"]] = 0

    return labels


def _reference_answers_for_output(rec: dict) -> list[str]:
    """Return whichever references are available for output scoring."""
    gold_answers = rec.get("gold_answers", [])
    if gold_answers:
        return gold_answers
    return rec.get("reference_answers", [])


# ── Per-strategy evaluation ────────────────────────────────────────────────
def evaluate_strategy(
    strategy:            str,
    gold_labels:         dict[str, int],
    always_rag_records:  Optional[list[dict]] = None,
    never_rag_records:   Optional[list[dict]] = None,
    verbose:             bool = False,
) -> dict:
    records = load_predictions(strategy)
    if not records:
        return {}

    f1_scores, recall_scores, rr_scores = [], [], []
    preds_for_bert, refs_for_bert       = [], []
    latencies                           = []
    routing_latencies                   = []
    retrieval_latencies                 = []
    n_retrieved                         = 0
    n_effective_retrieved               = 0
    feedback_before_f1, feedback_after_f1 = [], []
    feedback_deltas                       = []
    feedback_actions: dict[str, int]      = {}
    n_feedback_records                    = 0
    n_feedback_triggered                  = 0
    n_feedback_changed                    = 0
    n_feedback_improved                   = 0
    n_feedback_worsened                   = 0
    n_feedback_unchanged                  = 0
    n_reretrieved                         = 0
    n_reretrieve_gold_opportunities       = 0
    n_reretrieve_gold_recovered           = 0

    for rec in records:
        gold_answers = rec.get("gold_answers", [])
        answer       = rec.get("answer", "")
        gold_doc     = rec.get("gold_doc_id", "")
        initial_retrieved = rec.get("passages_retrieved", [])
        retrieved    = rec.get("final_passages_retrieved") or initial_retrieved
        feedback_action = rec.get("feedback_action")
        if feedback_action:
            feedback_actions[feedback_action] = (
                feedback_actions.get(feedback_action, 0) + 1
            )

        # Response quality — SQAC samples only (have gold answers)
        if gold_answers:
            f1 = token_f1(answer, gold_answers)
            f1_scores.append(f1)
            preds_for_bert.append(answer)
            refs_for_bert.append(gold_answers[0])

        # Self-feedback impact — all samples with references and an initial answer
        refs = _reference_answers_for_output(rec)
        initial_answer = rec.get("initial_answer")
        if refs and initial_answer is not None:
            before = token_f1(initial_answer, refs)
            after = token_f1(answer, refs)
            delta = after - before

            feedback_before_f1.append(before)
            feedback_after_f1.append(after)
            feedback_deltas.append(delta)
            n_feedback_records += 1

            if feedback_action in {
                "regenerate",
                "direct_fallback",
                "reretrieve",
                "reretrieve_regenerate",
            }:
                n_feedback_triggered += 1
            if initial_answer.strip() != answer.strip():
                n_feedback_changed += 1
            if delta > 1e-9:
                n_feedback_improved += 1
            elif delta < -1e-9:
                n_feedback_worsened += 1
            else:
                n_feedback_unchanged += 1

        # Retrieval quality — only when retrieval was triggered
        if rec.get("effective_retrieved", False) and retrieved and gold_doc:
            recall_scores.append(recall_at_3(retrieved, gold_doc))
            rr_scores.append(reciprocal_rank(retrieved, gold_doc))

        if rec.get("reretrieved", False):
            n_reretrieved += 1
            if gold_doc and gold_doc not in initial_retrieved[:3]:
                n_reretrieve_gold_opportunities += 1
                if gold_doc in retrieved[:3]:
                    n_reretrieve_gold_recovered += 1

        # Retrieval rate
        if rec.get("do_retrieve", False):
            n_retrieved += 1
        if rec.get("effective_retrieved", rec.get("do_retrieve", False)):
            n_effective_retrieved += 1

        latencies.append(rec.get("total_latency_s", 0.0))
        routing_latencies.append(rec.get("routing_latency_s", 0.0))
        retrieval_latencies.append(rec.get("retrieval_latency_s", 0.0))

        if verbose:
            f1_display = f"{f1_scores[-1]:.2f}" if gold_answers else "N/A"
            print(
                f"  [{rec.get('sample_id', '')}] "
                f"F1={f1_display}  "
                f"fb={feedback_action or 'N/A'}  "
                f"dec={rec.get('router_decision', '')}  "
                f"lat={latencies[-1]:.2f}s"
            )

    # BERTScore
    bert_scores = (
        compute_bertscore(preds_for_bert, refs_for_bert)
        if preds_for_bert else []
    )

    # Retrieval-decision classification metrics (all test samples).
    # Pre-feedback is the router's original retrieve/skip choice. Post-feedback
    # is the effective mode after the self-feedback loop can discard irrelevant
    # context or accept re-retrieved passages.
    pre_feedback_decision_met = retrieval_decision_classification_metrics(
        records,
        gold_labels,
        decision_key="do_retrieve",
    )
    post_feedback_decision_met = retrieval_decision_classification_metrics(
        records,
        gold_labels,
        decision_key="effective_retrieved",
    )

    # Oracle retrieval-decision evaluation. The oracle is derived at evaluation
    # time by comparing always_rag and never_rag BERTScore against gold answers.
    oracle_met: dict = {}
    if (
        always_rag_records is not None
        and never_rag_records is not None
    ):
        pre_oracle_met = compute_oracle_policy_accuracy(
            records,
            always_rag_records,
            never_rag_records,
            decision_key="do_retrieve",
            prefix="pre_feedback_oracle_policy",
        )
        post_oracle_met = compute_oracle_policy_accuracy(
            records,
            always_rag_records,
            never_rag_records,
            decision_key="effective_retrieved",
            prefix="post_feedback_oracle_policy",
        )
        oracle_met = {
            **pre_oracle_met,
            **post_oracle_met,
        }

    token_f1_pct = float(np.mean(f1_scores) * 100) if f1_scores else None
    bertscore_f1_pct = (
        float(np.mean(bert_scores) * 100) if bert_scores else None
    )
    feedback_before_pct = (
        float(np.mean(feedback_before_f1) * 100)
        if feedback_before_f1 else None
    )
    feedback_after_pct = (
        float(np.mean(feedback_after_f1) * 100)
        if feedback_after_f1 else None
    )
    feedback_delta_pct = (
        float(np.mean(feedback_deltas) * 100) if feedback_deltas else None
    )

    return {
        "strategy":          strategy,
        "n_sqac_samples":    len(f1_scores),
        "n_total_samples":   len(records),
        # Response quality
        "token_f1":          token_f1_pct,
        "bertscore_f1":      bertscore_f1_pct,
        # Retrieval quality
        "recall_at_3": (
            float(np.mean(recall_scores) * 100) if recall_scores else None
        ),
        "mrr": (
            float(np.mean(rr_scores) * 100) if rr_scores else None
        ),
        # Latency
        "mean_latency_s":    float(np.mean(latencies))     if latencies     else None,
        "mean_routing_latency_s": (
            float(np.mean(routing_latencies)) if routing_latencies else None
        ),
        "mean_retrieval_latency_s": (
            float(np.mean(retrieval_latencies)) if retrieval_latencies else None
        ),
        "retrieval_rate": (
            (n_retrieved / len(records)) * 100 if records else None
        ),
        "pre_feedback_retrieval_rate": (
            (n_retrieved / len(records)) * 100 if records else None
        ),
        "post_feedback_retrieval_rate": (
            (n_effective_retrieved / len(records)) * 100 if records else None
        ),
        # Self-feedback impact. These are None for old prediction files that
        # do not contain initial_answer/feedback_action yet.
        "feedback_n_scored": n_feedback_records,
        "feedback_token_f1_before": feedback_before_pct,
        "feedback_token_f1_after": feedback_after_pct,
        "feedback_token_f1_delta": feedback_delta_pct,
        "feedback_trigger_rate": (
            (n_feedback_triggered / n_feedback_records) * 100
            if n_feedback_records else None
        ),
        "feedback_changed_rate": (
            (n_feedback_changed / n_feedback_records) * 100
            if n_feedback_records else None
        ),
        "feedback_improved_rate": (
            (n_feedback_improved / n_feedback_records) * 100
            if n_feedback_records else None
        ),
        "feedback_worsened_rate": (
            (n_feedback_worsened / n_feedback_records) * 100
            if n_feedback_records else None
        ),
        "feedback_unchanged_rate": (
            (n_feedback_unchanged / n_feedback_records) * 100
            if n_feedback_records else None
        ),
        "feedback_actions": feedback_actions,
        "reretrieve_rate": (
            (n_reretrieved / len(records)) * 100 if records else None
        ),
        "reretrieve_gold_recovery_rate": (
            (
                n_reretrieve_gold_recovered
                / n_reretrieve_gold_opportunities
            ) * 100
            if n_reretrieve_gold_opportunities else None
        ),
        "reretrieve_gold_opportunities": n_reretrieve_gold_opportunities,
        # Backwards-compatible aliases for the pre-feedback router decision.
        **{
            f"retrieval_decision_{k}": v
            for k, v in pre_feedback_decision_met.items()
            if k != "report"
        },
        "retrieval_decision_report": pre_feedback_decision_met.get("report", ""),
        # Explicit pre/post self-feedback routing evaluations.
        **{
            f"pre_feedback_retrieval_decision_{k}": v
            for k, v in pre_feedback_decision_met.items()
            if k != "report"
        },
        "pre_feedback_retrieval_decision_report": (
            pre_feedback_decision_met.get("report", "")
        ),
        **{
            f"post_feedback_retrieval_decision_{k}": v
            for k, v in post_feedback_decision_met.items()
            if k != "report"
        },
        "post_feedback_retrieval_decision_report": (
            post_feedback_decision_met.get("report", "")
        ),
        # Backwards-compatible aliases for pre-feedback oracle policy accuracy.
        "oracle_policy_accuracy": oracle_met.get(
            "pre_feedback_oracle_policy_accuracy"
        ),
        "n_samples": oracle_met.get("pre_feedback_oracle_policy_n_samples"),
        "oracle_retrieve_rate": oracle_met.get(
            "pre_feedback_oracle_policy_retrieve_rate"
        ),
        "oracle_score_metric": oracle_met.get(
            "pre_feedback_oracle_policy_score_metric"
        ),
        # Explicit pre/post self-feedback oracle retrieval policy evaluation.
        **oracle_met,
    }


# ── Summary table ──────────────────────────────────────────────────────────
def print_summary_table(results: list[dict]) -> None:
    metrics = [
        ("Token F1",              "token_f1"),
        ("BERTScore F1",          "bertscore_f1"),
        ("Recall@3",              "recall_at_3"),
        ("MRR",                   "mrr"),
        ("Mean latency (s)",      "mean_latency_s"),
        ("Mean decision latency (s)", "mean_routing_latency_s"),
        ("Mean retrieval latency (s)", "mean_retrieval_latency_s"),
        ("Retrieval rate (pre-FB)", "pre_feedback_retrieval_rate"),
        ("Retrieval rate (post-FB)", "post_feedback_retrieval_rate"),
        ("FB scored samples",     "feedback_n_scored"),
        ("FB F1 before",          "feedback_token_f1_before"),
        ("FB F1 after",           "feedback_token_f1_after"),
        ("FB F1 delta",           "feedback_token_f1_delta"),
        ("FB trigger rate",       "feedback_trigger_rate"),
        ("FB changed rate",       "feedback_changed_rate"),
        ("FB improved rate",      "feedback_improved_rate"),
        ("FB worsened rate",      "feedback_worsened_rate"),
        ("Reretrieve rate",       "reretrieve_rate"),
        ("Reretrieve gold recovery", "reretrieve_gold_recovery_rate"),
        ("Pre-FB decision accuracy", "pre_feedback_retrieval_decision_accuracy"),
        ("Pre-FB decision macro F1", "pre_feedback_retrieval_decision_macro_f1"),
        ("Post-FB decision accuracy", "post_feedback_retrieval_decision_accuracy"),
        ("Post-FB decision macro F1", "post_feedback_retrieval_decision_macro_f1"),
        ("Pre-FB oracle accuracy", "pre_feedback_oracle_policy_accuracy"),
        ("Post-FB oracle accuracy", "post_feedback_oracle_policy_accuracy"),
        ("Oracle retrieve rate",  "pre_feedback_oracle_policy_retrieve_rate"),
    ]

    strats = [r["strategy"] for r in results]
    col_w  = 28
    val_w  = max(16, max((len(s) for s in strats), default=14) + 2)

    sep    = "─" * (col_w + val_w * len(strats))
    header = f"{'Metric':<{col_w}}" + "".join(f"{s:>{val_w}}" for s in strats)

    print("\n" + sep)
    print("  Evaluation Results — Unified Test Set")
    print(sep)
    print(header)
    print(sep)

    for label, key in metrics:
        row = f"{label:<{col_w}}"
        for r in results:
            val = r.get(key)
            row += f"{('N/A' if val is None else f'{val:.4f}'):>{val_w}}"
        print(row)

    print(sep + "\n")

    # Print per-strategy retrieval-decision classification reports
    for r in results:
        report = r.get("pre_feedback_retrieval_decision_report", "")
        if report:
            print(
                f"── Pre-feedback retrieval decision report: "
                f"{r['strategy']} ──"
            )
            print(report)
        post_report = r.get("post_feedback_retrieval_decision_report", "")
        if post_report:
            print(
                f"── Post-feedback effective retrieval decision report: "
                f"{r['strategy']} ──"
            )
            print(post_report)


# ── Save report ────────────────────────────────────────────────────────────
def save_report(results: list[dict]) -> None:
    # Remove non-serialisable report strings before saving
    saveable = []
    for r in results:
        r2 = {
            k: v for k, v in r.items()
            if k not in {
                "retrieval_decision_report",
                "pre_feedback_retrieval_decision_report",
                "post_feedback_retrieval_decision_report",
            }
        }
        saveable.append(r2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(saveable, f, ensure_ascii=False, indent=2)
    print(f"[Evaluate] Report saved → {REPORT_PATH}")


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline on the unified test set."
    )
    parser.add_argument(
        "--strategy",
        default="all",
        help=(
            "Prediction file suffix to evaluate, e.g. router_rag_setfit_15036. "
            "Use 'all' for discovered predictions, or a comma-separated list "
            "for an explicit multi-strategy evaluation."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample breakdown.",
    )
    return parser.parse_args()


def main() -> None:
    args         = _parse_args()
    gold_labels  = load_test_labels()

    if not gold_labels:
        print(
            "[Evaluate] Test set not found. "
            "Run load_sqac.py and load_chitchat.py first."
        )
        return

    print(f"[Evaluate] Test set: {len(gold_labels)} samples  "
          f"({sum(gold_labels.values())} label=1 / "
          f"{list(gold_labels.values()).count(0)} label=0)")

    if args.strategy == "all":
        discovered = sorted(
            p.stem.removeprefix("predictions_")
            for p in Path(OUTPUT_DIR).glob("predictions_*.jsonl")
        )
        preferred = [
            "always_rag",
            "never_rag",
            "router_rag_frozen_lr_15036",
            "router_rag_setfit_15036",
            "router_rag_finetune_12500",
            "router_rag",
        ]
        strategies = [
            s for s in preferred if s in discovered
        ] + [
            s for s in discovered if s not in preferred
        ]
    elif "," in args.strategy:
        strategies = [
            s.strip()
            for s in args.strategy.split(",")
            if s.strip()
        ]
    else:
        strategies = [args.strategy]

    # Pre-load always_rag and never_rag for oracle policy evaluation
    always_rag_records = load_predictions("always_rag")
    never_rag_records  = load_predictions("never_rag")

    all_results = []
    for strat in strategies:
        print(f"\n[Evaluate] Evaluating strategy: {strat} …")
        res = evaluate_strategy(
            strategy           = strat,
            gold_labels        = gold_labels,
            always_rag_records = always_rag_records,
            never_rag_records  = never_rag_records,
            verbose            = args.verbose,
        )
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
