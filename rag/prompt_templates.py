"""
rag/prompt_templates.py
Prompt templates for the Phase 3 RAG chain.

Separation of concerns: keeping prompts here makes them easy to iterate
on without touching the chain logic.

Two templates
-------------
SYSTEM_PROMPT   — Tells the LLM its role and constraints.
rag_user_prompt — Formats the retrieved context + the user's question
                  into a single user-turn prompt string.
"""

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a news assistant with access to recent news summaries.
Your job is to answer the user's question using ONLY the provided news context.

Rules:
- Answer factually and concisely.
- If the context does not contain enough information to answer, say so clearly.
- Do not invent facts or reference knowledge outside the provided summaries.
- If multiple topics are relevant, synthesise them into a coherent answer.
- Keep your answer under 200 words unless the user asks for more detail."""


# ── User prompt builder ───────────────────────────────────────────────────────

def rag_user_prompt(query: str, context_clusters: list[dict]) -> str:
    """
    Build the user-turn prompt by injecting retrieved cluster summaries
    as numbered context blocks.

    Args:
        query           : The user's natural-language question.
        context_clusters: List of cluster dicts from vector_store.query_clusters().
                          Each must have 'label' and 'summary' keys.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    if not context_clusters:
        context_block = "(No relevant news summaries found.)"
    else:
        lines = []
        for i, c in enumerate(context_clusters, 1):
            label   = c.get("label", "Unknown Topic")
            summary = c.get("summary", "").strip()
            similarity = c.get("similarity", 0)
            lines.append(
                f"[{i}] Topic: {label}  (relevance: {similarity:.0%})\n"
                f"    {summary}"
            )
        context_block = "\n\n".join(lines)

    return (
        f"CONTEXT (recent news summaries):\n"
        f"{context_block}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"Answer:"
    )


# ── Fallback prompt (no vector DB results) ────────────────────────────────────

def no_context_prompt(query: str) -> str:
    """Prompt used when the vector DB is empty or returns nothing."""
    return (
        f"The user asked: {query}\n\n"
        "There are no news summaries available in the database yet. "
        "Politely inform the user and suggest running the ingestion + "
        "processing pipeline to populate the news database."
    )
