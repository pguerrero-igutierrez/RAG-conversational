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
ROOT_DIR      = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR    = ROOT_DIR / "outputs"
TEST_SQAC_PATH     = PROCESSED_DIR / "test_sqac.jsonl"
TEST_CHITCHAT_PATH = PROCESSED_DIR / "test_chitchat.jsonl"
PRED_PATH_TPL = str(OUTPUT_DIR / "predictions_{run_name}.jsonl")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _log_progress(message: str) -> None:
    print(message, flush=True)

# ── Lazy imports (avoid loading GPU models until needed) ───────────────────
def _load_components(
    strategy: str,
    router_approach: str | None = None,
    router_size: int | None = None,
    router_model_dir: str | None = None,
):
    """
    Instantiate and return (llm_chat_fn, retriever_fn, router) according to
    the chosen strategy.  Models are loaded once and reused for the session.
    """
    from load_llm import llm_chat  # type: ignore

    retrieve = None
    if strategy != "never_rag":
        from retriever import retrieve  # type: ignore

    router = None
    if strategy == "router_rag":
        from router import QueryRouter  # type: ignore
        router = QueryRouter(
            approach=router_approach,
            size=router_size,
            model_dir=router_model_dir,
        )
    return llm_chat, retrieve, router


def _run_name(
    strategy: str,
    router_approach: str | None = None,
    router_size: int | None = None,
    run_name: str | None = None,
) -> str:
    if run_name:
        return run_name
    if strategy != "router_rag" or router_approach is None:
        return strategy
    if router_size is None:
        return f"{strategy}_{router_approach}"
    return f"{strategy}_{router_approach}_{router_size}"


def _load_batch_samples() -> list[dict]:
    samples: list[dict] = []

    if TEST_SQAC_PATH.exists():
        with open(TEST_SQAC_PATH, encoding="utf-8") as f:
            samples.extend(json.loads(line) for line in f)

    if TEST_CHITCHAT_PATH.exists():
        with open(TEST_CHITCHAT_PATH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                samples.append({
                    "id": r["id"],
                    "question": r.get("text", ""),
                    "doc_id": "",
                    "answers": [],
                    "reference_answers": r.get("answers", []),
                    "label": 0,
                })

    return samples


# ── Core pipeline step ─────────────────────────────────────────────────────
def run_query(
    query:      str,
    strategy:   str,          # "always_rag" | "never_rag" | "router_rag"
    llm_chat_fn,
    retrieve_fn,
    router,
    top_k:      int  = 3,
    use_feedback: bool = True,
    corpus_expected: bool = False,
) -> dict:
    """
    Execute the full pipeline for a single *query* under the given *strategy*.

    Returns a result dict suitable for serialisation to JSONL.
    """
    from generator import generate_answer   # type: ignore

    t_start = time.perf_counter()

    # ── Routing decision ──────────────────────────────────────────────────
    t_route = time.perf_counter()
    if strategy == "always_rag":
        do_retrieve = True
        router_label = "always"
    elif strategy == "never_rag":
        do_retrieve = False
        router_label = "never"
    else:                                   # router_rag
        do_retrieve  = router.needs_retrieval(query)
        router_label = "retrieve" if do_retrieve else "skip"
    routing_latency = time.perf_counter() - t_route

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
        retrieve_fn    = retrieve_fn if do_retrieve else None,
        use_self_feedback = use_feedback and do_retrieve,
        allow_reretrieve = corpus_expected,
        top_k          = top_k,
    )

    total_latency = time.perf_counter() - t_start

    return {
        "query":              query,
        "strategy":           strategy,
        "router_decision":    router_label,
        "do_retrieve":        do_retrieve,
        "passages_retrieved": [p["id"] for p in passages],
        "answer":             gen_out["answer"],
        "initial_answer":     gen_out.get("initial_answer", gen_out["answer"]),
        "feedback_action":    gen_out.get("feedback_action", "skipped"),
        "rewritten_query":    gen_out.get("rewritten_query", ""),
        "reretrieved":        gen_out.get("reretrieved", False),
        "reretrieval_latency_s": gen_out.get("reretrieval_latency_s", 0.0),
        "final_passages_retrieved": [
            p["id"] for p in gen_out.get("passages_used", [])
        ],
        "effective_retrieved":gen_out["retrieved"],
        "regenerated":        gen_out["regenerated"],
        "critique":           gen_out.get("critique", {}),
        "reretrieve_critique": gen_out.get("reretrieve_critique", {}),
        "routing_latency_s":  round(routing_latency, 4),
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
    run_name: str,
) -> str:
    """
    Run the pipeline over the entire validation set and save predictions.

    Returns the path of the written predictions file.
    """
    samples = _load_batch_samples()
    if not samples:
        raise FileNotFoundError(
            "Unified test set not found. Run load_sqac.py and "
            "load_chatsubs.py first."
        )

    pred_path = PRED_PATH_TPL.format(run_name=run_name)

    print(
        f"\n[Main] Batch mode | strategy={strategy} "
        f"| run={run_name} | {len(samples)} samples",
        flush=True,
    )
    print(f"[Main] Predictions → {pred_path}\n", flush=True)

    with open(pred_path, "w", encoding="utf-8") as out_f:
        batch_started_at = time.perf_counter()
        for i, sample in enumerate(samples, 1):
            query    = sample["question"]
            doc_id   = sample.get("doc_id", "")
            gold     = sample.get("answers", [])

            _log_progress(
                f"  [{i}/{len(samples)}] starting "
                f"id={sample.get('id', f's{i}')} | strategy={strategy}"
            )

            result = run_query(
                query       = query,
                strategy    = strategy,
                llm_chat_fn = llm_chat_fn,
                retrieve_fn = retrieve_fn,
                router      = router,
                top_k       = top_k,
                use_feedback= use_feedback,
                corpus_expected = bool(doc_id),
            )

            record = {
                **result,
                "run_name":        run_name,
                "sample_id":      sample.get("id", f"s{i}"),
                "gold_doc_id":    doc_id,
                "gold_answers":   gold,
                "reference_answers": sample.get("reference_answers", []),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            elapsed = time.perf_counter() - batch_started_at
            avg_per_sample = elapsed / i
            eta = avg_per_sample * (len(samples) - i)
            _log_progress(
                f"  [{i}/{len(samples)}] done | "
                f"decision={result['router_decision']} | "
                f"lat={result['total_latency_s']:.2f}s | "
                f"elapsed={_format_duration(elapsed)} | "
                f"ETA={_format_duration(eta)}"
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
    parser.add_argument(
        "--router_approach",
        choices=["frozen_lr", "setfit", "finetune"],
        default=None,
        help="Force a specific trained router approach for router_rag.",
    )
    parser.add_argument(
        "--router_size", type=int, default=None,
        help="Force a specific trained router size for router_rag.",
    )
    parser.add_argument(
        "--router_model_dir", type=str, default=None,
        help=(
            "Load a router directly from this model directory. "
            "Useful for oracle-trained SetFit checkpoints."
        ),
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help=(
            "Prediction-file suffix. Defaults to the strategy/model name; "
            "set this when using --router_model_dir to avoid overwrites."
        ),
    )
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────
def main() -> None:
    args = parser = _parse_args()
    use_feedback = not args.no_feedback
    run_name = _run_name(
        args.strategy,
        args.router_approach,
        args.router_size,
        args.run_name,
    )

    print("[Main] Loading models …")
    llm_chat_fn, retrieve_fn, router = _load_components(
        args.strategy,
        router_approach=args.router_approach,
        router_size=args.router_size,
        router_model_dir=args.router_model_dir,
    )
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
            run_name    = run_name,
        )


if __name__ == "__main__":
    main()
