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

SYSTEM_PROMPT = """You are a helpful news assistant. Answer the user's question using ONLY the provided news context.

Guidelines:
1. Provide a factual and direct answer based on the provided summaries.
2. If the context does not contain relevant information to answer the user's specific question, state clearly that you couldn't find any news about that topic in the database.
3. Do not make up facts or use outside knowledge.
4. Keep answers brief (under 120 words)."""


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
    """Prompt used when the vector DB returns nothing."""
    from storage.database import article_count
    try:
        counts = article_count()
        total_articles = counts.get("total", 0)
    except Exception:
        total_articles = 0
        
    if total_articles == 0:
        return (
            f"The user asked: {query}\n\n"
            "There are no articles in the database. "
            "Politely inform the user that the database is currently empty and suggest running ingestion and processing to load some news."
        )
    else:
        return (
            f"The user asked: {query}\n\n"
            f"The database contains {total_articles} articles, but none of them are semantically relevant to their query. "
            "Politely inform the user that you couldn't find any matching news articles in the database for their query."
        )
