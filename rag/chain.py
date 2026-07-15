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

from config.settings import RAG_TOP_K, RAG_SCORE_THRESHOLD, OLLAMA_BASE_URL, OLLAMA_MODEL, GEMINI_API_KEY
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

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Send a system + user prompt to Gemini or Ollama and return the response text.

    We build a minimal instruction-following prompt by concatenating
    system and user messages.
    """
    if GEMINI_API_KEY:
        import requests
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 400
            }
        }
        logger.info("Calling Gemini LLM API for RAG chain...")
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            logger.error("Failed to parse Gemini response: %s", e)
            raise RuntimeError("Invalid response structure from Gemini API") from e
    else:
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

def get_cluster_field(cluster_id: int) -> str:
    """Classify the cluster into 'world', 'science', or 'tech' based on article sources."""
    from storage.database import get_connection, db_execute, fetch_row, fetch_rows
    import json
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            db_execute(cursor, "SELECT article_ids FROM clusters WHERE id = ?", (cluster_id,))
            row = fetch_row(cursor)
            if not row or not row.get("article_ids"):
                return "other"
            article_ids = json.loads(row["article_ids"])
            if not article_ids:
                return "other"
            placeholders = ",".join("?" * len(article_ids))
            db_execute(
                cursor,
                f"SELECT source FROM articles WHERE id IN ({placeholders})", 
                article_ids
            )
            article_rows = fetch_rows(cursor)
        
        sources = [r.get("source") for r in article_rows if r.get("source")]
        if not sources:
            return "other"
            
        tech_count = 0
        world_count = 0
        science_count = 0
        
        for s in sources:
            s_lower = s.lower()
            if "nasa" in s_lower:
                science_count += 1
            elif "world" in s_lower:
                world_count += 1
            else:
                tech_count += 1
                
        if world_count >= science_count and world_count >= tech_count:
            return "world"
        elif science_count >= world_count and science_count >= tech_count:
            return "science"
        else:
            return "tech"
    except Exception as exc:
        logger.warning("Failed to classify cluster %d field: %s", cluster_id, exc)
        return "other"


def diversify_results(candidates: list[dict], top_k: int) -> list[dict]:
    """Select top_k candidates, ensuring diversity across world, science, and tech news fields."""
    diversified = []
    field_counts = {"world": 0, "science": 0, "tech": 0, "other": 0}
    max_limit = max(1, top_k // 2)  # Allow at most 2 from each field for top_k=5
    
    remaining_candidates = []
    for c in candidates:
        field = get_cluster_field(c["id"])
        c["field"] = field
        if field_counts.get(field, 0) < max_limit:
            diversified.append(c)
            field_counts[field] = field_counts.get(field, 0) + 1
            if len(diversified) == top_k:
                break
        else:
            remaining_candidates.append(c)
            
    if len(diversified) < top_k:
        for c in remaining_candidates:
            diversified.append(c)
            if len(diversified) == top_k:
                break
                
    return diversified


def retrieve(
    query: str,
    top_k: int = RAG_TOP_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
) -> list[dict]:
    """
    Step 1: Retrieve the top-k most relevant cluster summaries, diversified by source field.

    Args:
        query          : User's natural-language question or keyword string.
        top_k          : Max number of clusters to retrieve.
        score_threshold: Minimum similarity score (0–1) to include a result.

    Returns:
        List of cluster dicts with keys: id, label, summary,
        article_count, created_at, similarity, field.
    """
    # Fetch a larger candidate pool to allow meaningful diversification
    fetch_k = max(top_k * 3, 15)
    candidates = query_clusters(query, top_k=fetch_k, score_threshold=score_threshold)
    
    if not candidates:
        return []
        
    diversified = diversify_results(candidates, top_k)
    logger.info("Retrieved %d diversified clusters for query: '%s'", len(diversified), query)
    return diversified



def generate(context_clusters: list[dict], query: str) -> str:
    """
    Step 2: Given retrieved context + the original query, call LLM (Gemini or Ollama)
    and return the generated answer.

    Handles the empty-context case gracefully (informs user).
    """
    if not context_clusters:
        logger.warning("No context clusters — using no-context fallback prompt.")
        user_prompt = no_context_prompt(query)
    else:
        user_prompt = rag_user_prompt(query, context_clusters)

    try:
        answer = _call_llm(SYSTEM_PROMPT, user_prompt)
    except RuntimeError as exc:
        logger.error("LLM generation failed: %s", exc)
        answer = (
            "Sorry, I couldn't generate an answer right now because the "
            "language model is unavailable. Please check your connection or LLM status."
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

def personalised_ask(
    query: str,
    user_id: str,
    top_k: int = RAG_TOP_K,
) -> RAGResult:
    """
    Full personalised RAG pipeline:
    personalised_retrieve → generate → RAGResult.

    Uses the user's cluster weight vector to bias results toward
    topics they've interacted with positively.

    Args:
        query   : Natural-language question.
        user_id : User identifier (string).
        top_k   : Number of clusters to retrieve after re-ranking.

    Returns:
        RAGResult — same structure as ask(), with personalised sources.

    Example:
        result = personalised_ask("AI regulation news", user_id="alice")
    """
    from personalization.retriever import personalised_retrieve

    logger.info("Personalised RAG ask: '%s'  user='%s'", query, user_id)

    clusters = personalised_retrieve(query, user_id=user_id, top_k=top_k)
    answer   = generate(clusters, query)

    return RAGResult(
        query=query,
        answer=answer,
        sources=clusters,
        retrieved_count=len(clusters),
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
