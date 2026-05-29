"""
Summarize pre/post self-feedback latency without modifying saved evaluations.

Pre-SF latency is read from the no-feedback rerun prediction files:
  outputs/predictions_latency_nofb_*.jsonl

Post-SF latency is read from the original saved prediction files:
  outputs/predictions_*.jsonl

The script writes a separate summary file:
  outputs/latency_pre_post_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SUMMARY_PATH = OUTPUTS / "latency_pre_post_summary.json"

RUNS = [
    ("Always RAG", "always_rag", "latency_nofb_always_rag"),
    ("Never RAG", "never_rag", "latency_nofb_never_rag"),
    (
        "Frozen LR router",
        "router_rag_frozen_lr_15036",
        "latency_nofb_router_rag_frozen_lr_15036",
    ),
    (
        "SetFit router",
        "router_rag_setfit_15036",
        "latency_nofb_router_rag_setfit_15036",
    ),
    (
        "Fine-tuned router",
        "router_rag_finetune_12500",
        "latency_nofb_router_rag_finetune_12500",
    ),
    (
        "Oracle SetFit-500",
        "router_rag_oracle_setfit_500",
        "latency_nofb_router_rag_oracle_setfit_500",
    ),
    (
        "Oracle SetFit-64",
        "router_rag_oracle_setfit_64",
        "latency_nofb_router_rag_oracle_setfit_64",
    ),
]


def load_records(run_name: str) -> list[dict]:
    path = OUTPUTS / f"predictions_{run_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mean_latency(records: list[dict]) -> float:
    values = [float(r.get("total_latency_s", 0.0)) for r in records]
    if not values:
        raise ValueError("No latency values found.")
    return sum(values) / len(values)


def main() -> None:
    rows = []
    for label, post_run, pre_run in RUNS:
        post_records = load_records(post_run)
        pre_records = load_records(pre_run)
        pre = mean_latency(pre_records)
        post = mean_latency(post_records)
        rows.append({
            "model": label,
            "pre_feedback_latency_s": round(pre, 4),
            "post_feedback_latency_s": round(post, 4),
            "delta_latency_s": round(post - pre, 4),
            "pre_n_samples": len(pre_records),
            "post_n_samples": len(post_records),
        })

    SUMMARY_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {SUMMARY_PATH}")
    print("Model & Pre-SF & Post-SF & Delta \\\\")
    for row in rows:
        print(
            f"{row['model']} & "
            f"{row['pre_feedback_latency_s']:.2f}s & "
            f"{row['post_feedback_latency_s']:.2f}s & "
            f"{row['delta_latency_s']:+.2f}s \\\\"
        )


if __name__ == "__main__":
    main()
