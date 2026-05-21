"""
processing/summarizer.py
Generates a label and summary for each cluster using Ollama + Phi-3 Mini.

Two prompts per cluster:
  1. Label prompt   — asks the model for a short 3-5 word topic label
                      e.g. "US Federal Reserve Policy"
  2. Summary prompt — asks for a 3-4 sentence neutral news summary
                      covering all articles in the cluster

Input to the LLM is just the titles + first 200 chars of each article body.
We keep the prompt compact to stay well within Phi-3 Mini's context window
and to avoid sending paywalled full-text to an LLM.

Ollama must be running locally:
    ollama serve
    ollama pull phi3:mini
"""

import json
import logging
import time

import requests

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_MAX_TOKENS

logger = logging.getLogger(__name__)

OLLAMA_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

# How many articles to include in the prompt (avoid huge prompts)
MAX_ARTICLES_IN_PROMPT = 8

# Retry settings
MAX_RETRIES   = 3
RETRY_DELAY_S = 2.0


# ── Ollama client ─────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str) -> str:
    """
    Send a prompt to Ollama and return the response text.
    Raises RuntimeError if Ollama is unreachable after retries.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": OLLAMA_MAX_TOKENS,
            "temperature": 0.3,    # low temp → factual, consistent summaries
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Cannot connect to Ollama. Is it running?\n"
                    "  ollama serve\n"
                    "  ollama pull phi3:mini"
                )
            logger.warning("Ollama unreachable, retrying (%d/%d)...", attempt, MAX_RETRIES)
            time.sleep(RETRY_DELAY_S)
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    return ""


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_article_block(articles: list[dict]) -> str:
    """Format articles into a compact numbered block for the prompt."""
    lines = []
    for i, art in enumerate(articles[:MAX_ARTICLES_IN_PROMPT], 1):
        title   = art.get("title", "").strip()
        snippet = art.get("body",  "").strip()[:200].replace("\n", " ")
        lines.append(f"{i}. {title}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def _label_prompt(article_block: str) -> str:
    return f"""You are a news editor. Read the following news articles and output ONLY a short topic label (3-5 words, title case) that best describes what they all have in common. Output nothing else — no explanation, no punctuation at the end.

Articles:
{article_block}

Topic label:"""


def _summary_prompt(article_block: str) -> str:
    return f"""You are a neutral news summarizer. Read the following related news articles and write a concise 3-4 sentence summary covering the key facts and developments. Be factual and objective. Do not repeat the same point twice.

Articles:
{article_block}

Summary:"""


# ── Public interface ──────────────────────────────────────────────────────────

def summarize_cluster(articles: list[dict]) -> tuple[str, str]:
    """
    Generate a (label, summary) tuple for a group of related articles.

    For single-article clusters (noise/singletons) we use the article
    title as the label and its body snippet as the summary — no LLM call
    needed, saving latency.

    Args:
        articles: List of article dicts from the DB.

    Returns:
        (label, summary) — both strings.
    """
    if not articles:
        return "Unknown", ""

    # Singleton — skip the LLM call
    if len(articles) == 1:
        art = articles[0]
        label   = art.get("title", "Uncategorised")[:60]
        summary = art.get("body",  "").strip()[:400] or art.get("title", "")
        return label, summary

    article_block = _build_article_block(articles)

    # Label
    try:
        label = _ollama_generate(_label_prompt(article_block))
        label = label.strip().strip('"').strip("'")
        if not label or len(label) > 80:
            label = articles[0].get("title", "News Cluster")[:60]
    except RuntimeError as exc:
        logger.error("Label generation failed: %s", exc)
        label = articles[0].get("title", "News Cluster")[:60]

    # Summary
    try:
        summary = _ollama_generate(_summary_prompt(article_block))
    except RuntimeError as exc:
        logger.error("Summary generation failed: %s", exc)
        # Fall back to concatenating snippets
        summary = " ".join(
            a.get("body", "").strip()[:150]
            for a in articles[:3]
            if a.get("body")
        )

    logger.debug("Cluster label: %s", label)
    return label, summary