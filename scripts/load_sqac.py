"""
load_sqac.py
------------
Loads the SQAC dataset and produces:

  1. Retrieval corpus        — all unique contexts from train.json + dev.json
                               → data/indexes/sqac_corpus.jsonl

  2. Router train split      — all SQAC questions from train.json (label=1)
                               → data/processed/router_train_sqac.jsonl
                               (15,036 questions — full training set)

  3. Router val split        — remaining dev.json questions after test
                               → data/processed/router_val_sqac.jsonl

  4. Unified test split      — 85 questions from dev.json (label=1)
                               → data/processed/test_sqac.jsonl
                               Used for: router eval, retrieval eval,
                               generation eval, and oracle routing.

  Validation uses cheap token-F1 overlap, so it can use the full remaining
  dev split. Test is kept small because oracle routing requires 2 full LLM
  passes per sample.

  Dev split usage:
    85 → unified test
    remaining 1,779 → router val

Usage
-----
  python scripts/load_sqac.py
  python scripts/load_sqac.py --corpus_only
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from tqdm.auto import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT_DIR / "corpus"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INDEX_DIR     = ROOT_DIR / "data" / "indexes"

TRAIN_PATH = RAW_DIR / "train.json"
DEV_PATH   = RAW_DIR / "dev.json"

CORPUS_PATH       = INDEX_DIR / "sqac_corpus.jsonl"
ROUTER_TRAIN_PATH = PROCESSED_DIR / "router_train_sqac.jsonl"
ROUTER_VAL_PATH   = PROCESSED_DIR / "router_val_sqac.jsonl"
TEST_PATH         = PROCESSED_DIR / "test_sqac.jsonl"

for d in [PROCESSED_DIR, INDEX_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Split sizes ────────────────────────────────────────────────────────────
TEST_SIZE   = 85
RANDOM_SEED = 42


# ── Parser ─────────────────────────────────────────────────────────────────
def parse_sqac(path: Path) -> tuple[dict[str, str], list[dict]]:
    """
    Parse a SQAC JSON file.

    Returns
    -------
    contexts  : dict mapping context_text → doc_id
    questions : list of answerable question dicts with keys:
                id, title, question, context, doc_id, answers, label
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    articles:  list       = raw.get("data", []) if isinstance(raw, dict) else raw
    contexts:  dict[str, str] = {}
    questions: list[dict]     = []

    for article in tqdm(articles, desc=f"Parsing {Path(path).name}"):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            ctx = para["context"]
            if ctx not in contexts:
                contexts[ctx] = f"doc_{len(contexts)}"
            doc_id = contexts[ctx]

            for qa in para.get("qas", []):
                answers   = qa.get("answers", [])
                ans_texts = [a["text"] for a in answers] if answers else []
                if not ans_texts:
                    continue
                questions.append({
                    "id":       qa["id"],
                    "title":    title,
                    "question": qa["question"],
                    "context":  ctx,
                    "doc_id":   doc_id,
                    "answers":  ans_texts,
                    "label":    1,
                })

    return contexts, questions


# ── Writer ─────────────────────────────────────────────────────────────────
def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────
def main(corpus_only: bool = False) -> None:
    random.seed(RANDOM_SEED)

    # ── 1. Parse train.json → corpus + full router train set ──────────────
    print("\n[SQAC] Parsing train.json …")
    train_contexts, train_questions = parse_sqac(TRAIN_PATH)
    print(f"  {len(train_contexts):,} unique contexts")
    print(f"  {len(train_questions):,} answerable questions")

    # Build corpus from all train contexts
    corpus_rows = [
        {"id": doc_id, "context": ctx}
        for ctx, doc_id in sorted(
            train_contexts.items(),
            key=lambda x: int(x[1].split("_")[1]),
        )
    ]

    if not corpus_only:
        # Use ALL train questions for router training
        write_jsonl(train_questions, ROUTER_TRAIN_PATH)
        print(f"\n[SQAC] Router train → {ROUTER_TRAIN_PATH}  "
              f"({len(train_questions):,} questions — full set)")

    # ── 2. Parse dev.json → val + test + extend corpus ────────────────────
    print("\n[SQAC] Parsing dev.json …")
    dev_contexts, dev_questions = parse_sqac(DEV_PATH)
    print(f"  {len(dev_contexts):,} unique contexts")
    print(f"  {len(dev_questions):,} answerable questions")

    # Add any new dev contexts to corpus
    new_ctx_count = 0
    offset = len(train_contexts)
    for ctx in dev_contexts:
        if ctx not in train_contexts:
            doc_id = f"doc_{offset + new_ctx_count}"
            corpus_rows.append({"id": doc_id, "context": ctx})
            new_ctx_count += 1
    if new_ctx_count:
        print(f"  Added {new_ctx_count} new dev contexts to corpus.")

    write_jsonl(corpus_rows, CORPUS_PATH)
    print(f"\n[SQAC] Corpus → {CORPUS_PATH}  ({len(corpus_rows):,} passages)")

    if not corpus_only:
        if len(dev_questions) <= TEST_SIZE:
            raise ValueError(
                f"Not enough dev questions: {len(dev_questions)} <= "
                f"{TEST_SIZE}"
            )

        random.shuffle(dev_questions)
        test_qs    = dev_questions[:TEST_SIZE]
        router_val = dev_questions[TEST_SIZE:]

        write_jsonl(router_val, ROUTER_VAL_PATH)
        write_jsonl(test_qs,    TEST_PATH)
        print(f"[SQAC] Router val   → {ROUTER_VAL_PATH}  ({len(router_val):,})")
        print(f"[SQAC] Unified test → {TEST_PATH}  ({len(test_qs):,})")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  SQAC data preparation complete")
    print("─" * 55)
    print(f"  Corpus       : {len(corpus_rows):,} passages")
    if not corpus_only:
        print(f"  Router train : {len(train_questions):,} questions  (label=1, full set)")
        print(f"  Router val   : {len(router_val):,} questions  (label=1)")
        print(f"  Unified test : {TEST_SIZE:,} questions  (label=1)")
        print("  Dev unused   : 0 questions")
    print("─" * 55 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SQAC corpus and splits.")
    parser.add_argument(
        "--corpus_only", action="store_true",
        help="Rebuild corpus only without resampling question splits.",
    )
    args = parser.parse_args()
    main(corpus_only=args.corpus_only)
