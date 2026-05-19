"""
Build a small oracle-labelled training subset.

For each SQAC train example, this script generates both counterfactual outputs:
  - always_rag  -> retrieve
  - never_rag   -> skip retrieval

It then scores both answers against the gold textual answer with BERTScore F1.
The oracle label is:
  - 1 if always_rag scores higher
  - 0 otherwise

The script stops once it has target_per_label examples for each oracle label.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CORPUS_DIR = ROOT_DIR / "corpus"
TRAIN_SQAC_PATH = PROCESSED_DIR / "router_train_sqac.jsonl"
DEFAULT_OUTPUT_PATH = CORPUS_DIR / "oracle_train_sqac_64_per_label.jsonl"

sys.path.insert(0, str(ROOT_DIR / "scripts"))

from evaluate import compute_bertscore  # noqa: E402
from main import _format_duration, _load_components, run_query  # noqa: E402


def _load_sqac_train(seed: int) -> list[dict]:
    with open(TRAIN_SQAC_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    records = [r for r in records if r.get("answers")]
    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def _load_existing(path: Path) -> tuple[set[str], dict[int, int]]:
    seen_ids: set[str] = set()
    counts = {0: 0, 1: 0}
    if not path.exists():
        return seen_ids, counts

    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            seen_ids.add(rec["sample_id"])
            label = int(rec["oracle_label"])
            counts[label] += 1
    return seen_ids, counts


def _score_chunk(chunk: list[dict]) -> tuple[list[float], list[float]]:
    refs = [rec["gold_answers"][0] for rec in chunk]
    always_answers = [rec["always_rag"]["answer"] for rec in chunk]
    never_answers = [rec["never_rag"]["answer"] for rec in chunk]
    return (
        compute_bertscore(always_answers, refs),
        compute_bertscore(never_answers, refs),
    )


def build_oracle_subset(
    target_per_label: int,
    candidate_batch: int,
    seed: int,
    top_k: int,
    output_path: Path,
    max_candidates: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids, counts = _load_existing(output_path)
    samples = _load_sqac_train(seed)

    if counts[0] >= target_per_label and counts[1] >= target_per_label:
        print(
            f"[Oracle] Target already satisfied in {output_path}: "
            f"skip={counts[0]} retrieve={counts[1]}",
            flush=True,
        )
        return

    if max_candidates == 0:
        print("[Oracle] max_candidates=0; exiting before model load.", flush=True)
        return

    print("[Oracle] Loading models …", flush=True)
    llm_chat_fn, retrieve_fn, _ = _load_components("always_rag")
    print("[Oracle] Components loaded.", flush=True)
    print(
        f"[Oracle] Target per label: {target_per_label} | "
        f"existing counts: skip={counts[0]} retrieve={counts[1]}",
        flush=True,
    )
    print(f"[Oracle] Output → {output_path}", flush=True)

    started_at = time.perf_counter()
    attempted = 0
    pending: list[dict] = []

    with open(output_path, "a", encoding="utf-8") as out_f:
        for sample in samples:
            if counts[0] >= target_per_label and counts[1] >= target_per_label:
                break
            if max_candidates is not None and attempted >= max_candidates:
                break
            if sample["id"] in seen_ids:
                continue

            attempted += 1
            query = sample["question"]
            doc_id = sample.get("doc_id", "")
            gold_answers = sample.get("answers", [])

            print(
                f"[Oracle] candidate {attempted} id={sample['id']} | "
                f"counts skip={counts[0]} retrieve={counts[1]}",
                flush=True,
            )

            always_out = run_query(
                query=query,
                strategy="always_rag",
                llm_chat_fn=llm_chat_fn,
                retrieve_fn=retrieve_fn,
                router=None,
                top_k=top_k,
                use_feedback=True,
                corpus_expected=bool(doc_id),
            )
            never_out = run_query(
                query=query,
                strategy="never_rag",
                llm_chat_fn=llm_chat_fn,
                retrieve_fn=retrieve_fn,
                router=None,
                top_k=top_k,
                use_feedback=False,
                corpus_expected=False,
            )

            pending.append({
                "sample_id": sample["id"],
                "query": query,
                "gold_doc_id": doc_id,
                "gold_answers": gold_answers,
                "source_label": sample.get("label"),
                "always_rag": always_out,
                "never_rag": never_out,
            })

            if len(pending) < candidate_batch:
                continue

            counts = _flush_scored_records(
                pending,
                counts,
                target_per_label,
                out_f,
            )
            pending = []

        if pending and (
            counts[0] < target_per_label or counts[1] < target_per_label
        ):
            counts = _flush_scored_records(
                pending,
                counts,
                target_per_label,
                out_f,
            )

    elapsed = time.perf_counter() - started_at
    print(
        f"[Oracle] Done | attempted={attempted} | "
        f"skip={counts[0]} retrieve={counts[1]} | "
        f"elapsed={_format_duration(elapsed)}",
        flush=True,
    )


def _flush_scored_records(
    records: list[dict],
    counts: dict[int, int],
    target_per_label: int,
    out_f,
) -> dict[int, int]:
    always_scores, never_scores = _score_chunk(records)
    for rec, always_score, never_score in zip(
        records,
        always_scores,
        never_scores,
    ):
        oracle_label = 1 if always_score > never_score else 0
        if counts[oracle_label] >= target_per_label:
            continue

        rec["oracle_label"] = oracle_label
        rec["oracle_decision"] = "retrieve" if oracle_label else "skip"
        rec["always_bertscore_f1"] = float(always_score)
        rec["never_bertscore_f1"] = float(never_score)
        rec["oracle_score_metric"] = "bertscore_f1"

        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        counts[oracle_label] += 1
        print(
            f"[Oracle] accepted label={oracle_label} | "
            f"skip={counts[0]} retrieve={counts[1]} | "
            f"always={always_score:.4f} never={never_score:.4f}",
            flush=True,
        )
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate oracle labels from train SQAC counterfactuals.",
    )
    parser.add_argument("--target_per_label", type=int, default=64)
    parser.add_argument("--candidate_batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max_candidates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_oracle_subset(
        target_per_label=args.target_per_label,
        candidate_batch=args.candidate_batch,
        seed=args.seed,
        top_k=args.top_k,
        output_path=args.output,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
