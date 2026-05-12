"""
main.py
-------
Main orchestrator for the RAG-based Conversational System with Query Router.

Pipeline
--------
  User query
      │
      ▼
  QueryRouter  ──► needs_retrieval?
      │                   │
    [Yes]               [No]
      │                   │
  Retriever           (skip)
      │                   │
      └────────┬──────────┘
               ▼
           Generator  (+ optional self-feedback loop)
               │
               ▼
          Final Answer

Modes
-----
  --mode interactive   : REPL loop — type queries one at a time.
  --mode batch         : Run the full evaluation set through the pipeline
                         and write predictions to a JSONL file for evaluate.py.
  --mode single        : Run a single query passed via --query.

Flags
-----
  --strategy  always_rag | never_rag | router_rag   (default: router_rag)
  --top_k     Number of passages to retrieve        (default: 3)
  --no_feedback   Disable the self-feedback loop
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────
BASE_DIR      = "./rag_project"
INDEX_DIR     = f"{BASE_DIR}/indexes"
DATA_DIR      = f"{BASE_DIR}/data"
OUTPUT_DIR    = f"{BASE_DIR}/outputs"
ROUTER_DIR    = f"{BASE_DIR}/router_model"
EVAL_PATH     = f"{DATA_DIR}/sqac_validation.jsonl"
PRED_PATH_TPL = f"{OUTPUT_DIR}/predictions_{{strategy}}.jsonl"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Lazy imports (avoid loading GPU models until needed) ───────────────────
def _load_components(strategy: str):
    """
    Instantiate and return (llm_chat_fn, retriever_fn, router) according to
    the chosen strategy.  Models are loaded once and reused for the session.
    """
    from load_llm  import llm_chat          # type: ignore
    from retriever import retrieve           # type: ignore
    from router    import QueryRouter        # type: ignore

    router = QueryRouter(model_dir=ROUTER_DIR, use_setfit=True)
    return llm_chat, retrieve, router


# ── Core pipeline step ─────────────────────────────────────────────────────
def run_query(
    query:      str,
    strategy:   str,          # "always_rag" | "never_rag" | "router_rag"
    llm_chat_fn,
    retrieve_fn,
    router,
    top_k:      int  = 3,
    use_feedback: bool = True,
) -> dict:
    """
    Execute the full pipeline for a single *query* under the given *strategy*.

    Returns a result dict suitable for serialisation to JSONL.
    """
    from generator import generate_answer   # type: ignore

    t_start = time.perf_counter()

    # ── Routing decision ──────────────────────────────────────────────────
    if strategy == "always_rag":
        do_retrieve = True
        router_label = "always"
    elif strategy == "never_rag":
        do_retrieve = False
        router_label = "never"
    else:                                   # router_rag
        do_retrieve  = router.needs_retrieval(query)
        router_label = "retrieve" if do_retrieve else "skip"

    # ── Retrieval ─────────────────────────────────────────────────────────
    passages: list[dict] = []
    retrieval_latency = 0.0
    if do_retrieve:
        t_r = time.perf_counter()
        passages = retrieve_fn(query, top_k=top_k)
        retrieval_latency = time.perf_counter() - t_r

    # ── Generation ────────────────────────────────────────────────────────
    gen_out = generate_answer(
        query          = query,
        passages       = passages if do_retrieve else None,
        llm_chat_fn    = llm_chat_fn,
        use_self_feedback = use_feedback and do_retrieve,
    )

    total_latency = time.perf_counter() - t_start

    return {
        "query":              query,
        "strategy":           strategy,
        "router_decision":    router_label,
        "do_retrieve":        do_retrieve,
        "passages_retrieved": [p["id"] for p in passages],
        "answer":             gen_out["answer"],
        "effective_retrieved":gen_out["retrieved"],
        "regenerated":        gen_out["regenerated"],
        "critique":           gen_out.get("critique", {}),
        "retrieval_latency_s":round(retrieval_latency, 4),
        "total_latency_s":    round(total_latency, 4),
    }


# ── Interactive REPL ───────────────────────────────────────────────────────
def interactive_mode(
    strategy: str,
    llm_chat_fn,
    retrieve_fn,
    router,
    top_k: int,
    use_feedback: bool,
) -> None:
    print("\n" + "═" * 60)
    print("  RAG Conversational System — Interactive Mode")
    print(f"  Strategy : {strategy}")
    print("  Type 'salir' or 'exit' to quit.")
    print("═" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if query.lower() in {"salir", "exit", "quit", "q"}:
            print("[Session ended]")
            break
        if not query:
            continue

        result = run_query(
            query       = query,
            strategy    = strategy,
            llm_chat_fn = llm_chat_fn,
            retrieve_fn = retrieve_fn,
            router      = router,
            top_k       = top_k,
            use_feedback= use_feedback,
        )

        # Pretty print
        tag = f"[{result['router_decision'].upper()}]"
        print(f"\nAssistant {tag}: {result['answer']}")
        print(
            f"  ↳ latency={result['total_latency_s']:.2f}s | "
            f"retrieved={result['effective_retrieved']} | "
            f"regenerated={result['regenerated']}\n"
        )


# ── Batch mode (writes predictions for evaluate.py) ───────────────────────
def batch_mode(
    strategy: str,
    llm_chat_fn,
    retrieve_fn,
    router,
    top_k: int,
    use_feedback: bool,
    eval_path: str = EVAL_PATH,
) -> str:
    """
    Run the pipeline over the entire validation set and save predictions.

    Returns the path of the written predictions file.
    """
    if not Path(eval_path).exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {eval_path}\n"
            "Run load_dataset.py first."
        )

    with open(eval_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]

    pred_path = PRED_PATH_TPL.format(strategy=strategy)

    print(f"\n[Main] Batch mode | strategy={strategy} | {len(samples)} samples")
    print(f"[Main] Predictions → {pred_path}\n")

    with open(pred_path, "w", encoding="utf-8") as out_f:
        for i, sample in enumerate(samples, 1):
            query    = sample["question"]
            doc_id   = sample.get("doc_id", "")
            gold     = sample.get("answers", [])

            result = run_query(
                query       = query,
                strategy    = strategy,
                llm_chat_fn = llm_chat_fn,
                retrieve_fn = retrieve_fn,
                router      = router,
                top_k       = top_k,
                use_feedback= use_feedback,
            )

            record = {
                **result,
                "sample_id":      sample.get("id", f"s{i}"),
                "gold_doc_id":    doc_id,
                "gold_answers":   gold,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i % 10 == 0 or i == len(samples):
                print(
                    f"  [{i}/{len(samples)}] "
                    f"decision={result['router_decision']} | "
                    f"lat={result['total_latency_s']:.2f}s"
                )

    print(f"\n[Main] Done. Predictions saved → {pred_path}")
    return pred_path


# ── Argument parsing ───────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Conversational System with Query Router"
    )
    parser.add_argument(
        "--mode", choices=["interactive", "batch", "single"],
        default="interactive",
        help="Run mode (default: interactive).",
    )
    parser.add_argument(
        "--strategy",
        choices=["always_rag", "never_rag", "router_rag"],
        default="router_rag",
        help="Retrieval strategy (default: router_rag).",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query string for --mode single.",
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Number of passages to retrieve (default: 3).",
    )
    parser.add_argument(
        "--no_feedback", action="store_true",
        help="Disable the self-feedback verification loop.",
    )
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────
def main() -> None:
    args = parser = _parse_args()
    use_feedback = not args.no_feedback

    print("[Main] Loading models …")
    llm_chat_fn, retrieve_fn, router = _load_components(args.strategy)
    print("[Main] All components loaded.\n")

    if args.mode == "interactive":
        interactive_mode(
            strategy    = args.strategy,
            llm_chat_fn = llm_chat_fn,
            retrieve_fn = retrieve_fn,
            router      = router,
            top_k       = args.top_k,
            use_feedback= use_feedback,
        )

    elif args.mode == "single":
        if not args.query:
            raise ValueError("--mode single requires --query <question>.")
        result = run_query(
            query       = args.query,
            strategy    = args.strategy,
            llm_chat_fn = llm_chat_fn,
            retrieve_fn = retrieve_fn,
            router      = router,
            top_k       = args.top_k,
            use_feedback= use_feedback,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == "batch":
        batch_mode(
            strategy    = args.strategy,
            llm_chat_fn = llm_chat_fn,
            retrieve_fn = retrieve_fn,
            router      = router,
            top_k       = args.top_k,
            use_feedback= use_feedback,
        )


if __name__ == "__main__":
    main()
