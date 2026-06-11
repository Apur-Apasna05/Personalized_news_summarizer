"""
rag/chain.py
Phase 3 RAG pipeline — retrieve relevant cluster summaries, then
generate an answer using Ollama.

Flow
----
  query (str)
     │
     ▼
  vector_store.query_clusters()   ← embed query + cosine search in ChromaDB
     │  top-k cluster summaries
     ▼
  build prompt (system + context + question)
     │
     ▼
  Ollama (phi3:mini or configured model)
     │  natural-language answer
     ▼
  RAGResult(answer, sources, query)

Usage
-----
    from rag.chain import ask

    result = ask("What is happening with AI regulation?")
    print(result.answer)
    for src in result.sources:
        print(f"  • {src['label']} ({src['similarity']:.0%})")

You can also call the lower-level helpers directly:
    retrieved = retrieve("AI regulation")
    answer    = generate(retrieved, "AI regulation")
"""

import logging
from dataclasses import dataclass, field

from config.settings import RAG_TOP_K, RAG_SCORE_THRESHOLD, OLLAMA_BASE_URL, OLLAMA_MODEL
from storage.vector_store import query_clusters
from rag.prompt_templates import SYSTEM_PROMPT, rag_user_prompt, no_context_prompt

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """Structured result returned by ask()."""
    query:   str
    answer:  str
    sources: list[dict] = field(default_factory=list)
    # How many clusters were retrieved before score filtering
    retrieved_count: int = 0

    def pretty(self) -> str:
        """Human-readable formatted string for CLI/logging."""
        lines = [
            f"Q: {self.query}",
            f"\nA: {self.answer}",
        ]
        if self.sources:
            lines.append("\nSources:")
            for s in self.sources:
                lines.append(
                    f"  • [{s['similarity']:.0%}] {s['label']}"
                )
        return "\n".join(lines)


# ── Ollama call (reuse pattern from summarizer, keep DRY) ─────────────────────

def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """
    Send a system + user prompt to Ollama and return the response text.

    We build a minimal instruction-following prompt by concatenating
    system and user messages — Phi-3 Mini responds well to this format.
    Raises RuntimeError if Ollama is unreachable.
    """
    import time
    import requests

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": 400,    # slightly more room for answers than summaries
            "temperature": 0.4,
        },
    }

    endpoint = f"{OLLAMA_BASE_URL}/api/generate"
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(endpoint, json=payload, timeout=90)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            if attempt == max_retries:
                raise RuntimeError(
                    "Cannot connect to Ollama. Make sure it's running:\n"
                    "  ollama serve\n"
                    f"  ollama pull {OLLAMA_MODEL}"
                )
            logger.warning("Ollama unreachable, retrying (%d/%d)...", attempt, max_retries)
            time.sleep(2.0)
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    return ""


# ── RAG steps ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = RAG_TOP_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
) -> list[dict]:
    """
    Step 1: Retrieve the top-k most relevant cluster summaries.

    Args:
        query          : User's natural-language question or keyword string.
        top_k          : Max number of clusters to retrieve.
        score_threshold: Minimum similarity score (0–1) to include a result.

    Returns:
        List of cluster dicts with keys: id, label, summary,
        article_count, created_at, similarity.
    """
    clusters = query_clusters(query, top_k=top_k, score_threshold=score_threshold)
    logger.info("Retrieved %d clusters for query: '%s'", len(clusters), query)
    return clusters


def generate(context_clusters: list[dict], query: str) -> str:
    """
    Step 2: Given retrieved context + the original query, call Ollama
    and return the generated answer.

    Handles the empty-context case gracefully (informs user).
    """
    if not context_clusters:
        logger.warning("No context clusters — using no-context fallback prompt.")
        user_prompt = no_context_prompt(query)
    else:
        user_prompt = rag_user_prompt(query, context_clusters)

    try:
        answer = _call_ollama(SYSTEM_PROMPT, user_prompt)
    except RuntimeError as exc:
        logger.error("Ollama generation failed: %s", exc)
        answer = (
            "Sorry, I couldn't generate an answer right now because the "
            "language model is unavailable. Please ensure Ollama is running."
        )

    return answer


# ── High-level entry point ────────────────────────────────────────────────────

def ask(
    query: str,
    top_k: int = RAG_TOP_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
) -> RAGResult:
    """
    Full RAG pipeline: retrieve → generate → return structured result.

    Args:
        query          : Natural-language question.
        top_k          : Number of clusters to retrieve.
        score_threshold: Minimum similarity for a cluster to be included.

    Returns:
        RAGResult with .answer (str) and .sources (list of cluster dicts).

    Example:
        result = ask("What's the latest on inflation?")
        print(result.answer)
    """
    logger.info("RAG ask: '%s'", query)

    # Retrieve
    clusters = retrieve(query, top_k=top_k, score_threshold=score_threshold)
    retrieved_count = len(clusters)

    # Generate
    answer = generate(clusters, query)

    return RAGResult(
        query=query,
        answer=answer,
        sources=clusters,
        retrieved_count=retrieved_count,
    )


# ── Interactive test helper ───────────────────────────────────────────────────

def interactive_session():
    """
    Simple REPL for testing the RAG chain from the command line.
    Exit with Ctrl+C or by typing 'quit'.
    """
    print("=== News RAG — interactive mode ===")
    print("Type a question, or 'quit' to exit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        result = ask(query)
        print(f"\nAssistant: {result.answer}")
        if result.sources:
            print("\nSources used:")
            for s in result.sources:
                print(f"  • [{s['similarity']:.0%}] {s['label']}")
        print()


if __name__ == "__main__":
    interactive_session()
