"""
generator.py
------------
Handles prompt construction, answer generation, and the self-feedback
verification loop inspired by Self-RAG.

Responsibilities
----------------
1. Build a structured Spanish-language prompt from a query + (optional)
   retrieved passages.
2. Call the LLM via the llm_chat() helper from load_llm.py.
3. Run a lightweight self-feedback pass: ask the same LLM whether its own
   answer is supported by the retrieved context and, if not, trigger a
   second generation attempt (one retry).

Self-feedback Loop (simplified Self-RAG critique)
--------------------------------------------------
After generating an answer with retrieved context, the generator asks the
LLM to rate its own output on two axes:
  - ISREL  : Is the retrieved context relevant to the question?
  - ISSUP  : Is the answer fully supported by the retrieved context?

If either check fails the model is asked to regenerate with an explicit
instruction to rely only on the provided passages.  No external critic model
is needed — the same Mixtral instruction-tuned model acts as its own judge.
"""

from __future__ import annotations

import re
from typing import Optional
from config import HF_TOKEN

# ── Prompt templates ───────────────────────────────────────────────────────
_SYSTEM_RAG = (
    "Eres un asistente experto que responde preguntas en español de forma "
    "precisa y concisa. Utiliza ÚNICAMENTE la información de los fragmentos "
    "proporcionados para elaborar tu respuesta. Si los fragmentos no contienen "
    "la respuesta, indícalo claramente."
)

_SYSTEM_DIRECT = (
    "Eres un asistente experto que responde preguntas en español de forma "
    "precisa y concisa. Responde basándote en tu conocimiento general."
)

_SYSTEM_CRITIC = (
    "Eres un evaluador estricto de respuestas en español. Tu tarea es verificar "
    "si una respuesta está respaldada por los fragmentos de contexto dados. "
    "Responde SOLO con un JSON válido y nada más."
)

_SYSTEM_REGENERATE = (
    "Eres un asistente experto que responde preguntas en español. "
    "Tu respuesta anterior no estaba suficientemente respaldada por el contexto. "
    "Genera una nueva respuesta basándote ÚNICAMENTE en los fragmentos proporcionados. "
    "Si la información no está en los fragmentos, di: 'No lo sé con certeza.'"
)


def _format_passages(passages: list[dict]) -> str:
    """Format retrieved passage dicts into a numbered context block."""
    if not passages:
        return "(sin contexto)"
    parts = []
    for i, p in enumerate(passages, 1):
        title = p.get("title", "")
        ctx   = p.get("context", "")
        parts.append(f"[{i}] {title}\n{ctx}")
    return "\n\n".join(parts)


def build_rag_prompt(query: str, passages: list[dict]) -> tuple[str, str]:
    """
    Build a (system, user) prompt pair for RAG generation.

    Returns
    -------
    system : system-role string
    user   : user-role string containing context + question
    """
    context_block = _format_passages(passages)
    user = (
        f"Fragmentos de contexto:\n{context_block}\n\n"
        f"Pregunta: {query}\n\n"
        f"Respuesta:"
    )
    return _SYSTEM_RAG, user


def build_direct_prompt(query: str) -> tuple[str, str]:
    """Build a (system, user) prompt for direct (no retrieval) generation."""
    user = f"Pregunta: {query}\n\nRespuesta:"
    return _SYSTEM_DIRECT, user


# ── Self-feedback / critique ───────────────────────────────────────────────
def _parse_critique_json(raw: str) -> dict:
    """
    Extract JSON from the critic LLM output.
    Returns a dict with keys 'isrel' and 'issup' (both bool).
    Defaults to True (pass) on parse failure to avoid blocking the pipeline.
    """
    import json

    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    # Try direct parse first
    try:
        data = json.loads(raw)
        return {
            "isrel": bool(data.get("isrel", True)),
            "issup": bool(data.get("issup", True)),
        }
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    isrel = re.search(r'"isrel"\s*:\s*(true|false)', raw, re.I)
    issup = re.search(r'"issup"\s*:\s*(true|false)', raw, re.I)
    return {
        "isrel": isrel.group(1).lower() == "true" if isrel else True,
        "issup": issup.group(1).lower() == "true" if issup else True,
    }


def critique_answer(
    query: str,
    passages: list[dict],
    answer: str,
    llm_chat_fn,  # callable: (system, user, **kwargs) -> str
) -> dict:
    """
    Ask the LLM to evaluate its own answer against the retrieved passages.

    Returns
    -------
    dict with keys:
        isrel  (bool) : context is relevant to the question
        issup  (bool) : answer is supported by the context
        raw    (str)  : raw LLM output for debugging
    """
    context_block = _format_passages(passages)
    user = (
        f"Pregunta: {query}\n\n"
        f"Fragmentos de contexto:\n{context_block}\n\n"
        f"Respuesta generada: {answer}\n\n"
        "Evalúa la respuesta y devuelve SOLO este JSON:\n"
        '{"isrel": <true|false>, "issup": <true|false>}\n\n'
        "- isrel: ¿Los fragmentos son relevantes para la pregunta?\n"
        "- issup: ¿La respuesta está totalmente respaldada por los fragmentos?"
    )
    raw = llm_chat_fn(
        system=_SYSTEM_CRITIC,
        user=user,
        temperature=0.0,
        max_new_tokens=64,
    )
    result = _parse_critique_json(raw)
    result["raw"] = raw
    return result


# ── Main generation entry point ────────────────────────────────────────────
def generate_answer(
    query: str,
    passages: Optional[list[dict]],
    llm_chat_fn,                   # callable from load_llm.py
    use_self_feedback: bool = True,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
) -> dict:
    """
    Generate an answer for *query*, optionally using retrieved *passages*.

    Parameters
    ----------
    query            : The user's natural language question.
    passages         : List of retrieved passage dicts (may be None or []).
    llm_chat_fn      : The llm_chat() callable from load_llm.py.
    use_self_feedback: If True, run the critique + conditional regeneration
                       loop when passages are provided.
    temperature      : Sampling temperature for generation.
    max_new_tokens   : Maximum tokens to generate.

    Returns
    -------
    dict with keys:
        answer          (str)  : final answer string
        retrieved       (bool) : whether passages were used
        critique        (dict) : critique result (only when retrieved + feedback)
        regenerated     (bool) : whether a second generation was triggered
        passages_used   (list) : the passage dicts actually used
    """
    result: dict = {
        "answer":        "",
        "retrieved":     False,
        "critique":      {},
        "regenerated":   False,
        "passages_used": [],
    }

    # ── Path A: no retrieval ───────────────────────────────────────────────
    if not passages:
        system, user = build_direct_prompt(query)
        result["answer"]    = llm_chat_fn(
            system=system,
            user=user,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        result["retrieved"] = False
        return result

    # ── Path B: retrieval-augmented generation ─────────────────────────────
    result["retrieved"]     = True
    result["passages_used"] = passages

    system, user = build_rag_prompt(query, passages)
    answer = llm_chat_fn(
        system=system,
        user=user,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    # ── Self-feedback critique ─────────────────────────────────────────────
    if use_self_feedback:
        critique = critique_answer(query, passages, answer, llm_chat_fn)
        result["critique"] = critique

        # Regenerate if context is relevant but answer is not well supported
        if critique["isrel"] and not critique["issup"]:
            print("[Generator] Self-feedback: answer not supported. Regenerating …")
            system2 = _SYSTEM_REGENERATE
            _, user2 = build_rag_prompt(query, passages)
            answer = llm_chat_fn(
                system=system2,
                user=user2,
                temperature=0.0,          # greedy for safer factual answer
                max_new_tokens=max_new_tokens,
            )
            result["regenerated"] = True

        elif not critique["isrel"]:
            # Context is off-topic — fall back to direct generation
            print("[Generator] Self-feedback: context not relevant. "
                  "Falling back to direct generation …")
            system3, user3 = build_direct_prompt(query)
            answer = llm_chat_fn(
                system=system3,
                user=user3,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            result["retrieved"]     = False   # effective mode is direct
            result["passages_used"] = []

    result["answer"] = answer
    return result


# ── CLI smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal offline test using a mock llm_chat to verify prompt construction.
    def _mock_llm(system: str, user: str, **kwargs) -> str:  # noqa: ANN001
        print(f"\n[MOCK LLM]\nSYSTEM: {system[:80]}…\nUSER:\n{user[:300]}…\n")
        return "Respuesta de prueba generada por el mock."

    sample_passages = [
        {
            "id":      "doc_0",
            "title":   "Universidad de Salamanca",
            "context": "La Universidad de Salamanca fue fundada en 1218 por el rey "
                       "Alfonso IX de León, siendo una de las más antiguas de Europa.",
        }
    ]
    sample_query = "¿Cuándo se fundó la Universidad de Salamanca?"

    print("── RAG generation (with self-feedback) ──")
    out = generate_answer(
        query=sample_query,
        passages=sample_passages,
        llm_chat_fn=_mock_llm,
        use_self_feedback=True,
    )
    print(f"Answer     : {out['answer']}")
    print(f"Retrieved  : {out['retrieved']}")
    print(f"Regenerated: {out['regenerated']}")

    print("\n── Direct generation (no retrieval) ──")
    out2 = generate_answer(
        query="¿Cuánto es 12 por 12?",
        passages=None,
        llm_chat_fn=_mock_llm,
        use_self_feedback=False,
    )
    print(f"Answer    : {out2['answer']}")
    print(f"Retrieved : {out2['retrieved']}")
