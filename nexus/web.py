"""Web search shared by the chat agent and the Corrective-RAG fallback.

Uses Tavily when ``TAVILY_API_KEY`` is set (as in corrective-rag), otherwise
falls back to DuckDuckGo (as in chatbot-in-langgraph) so it works with no key.
"""
from __future__ import annotations

import logging

from langsmith import traceable

from .config import settings

log = logging.getLogger(__name__)


def provider_name() -> str:
    return "Tavily" if settings.tavily_api_key else "DuckDuckGo"


@traceable(name="web_search", run_type="tool")
def search(query: str, max_results: int = 5) -> list[dict]:
    """Return a list of ``{"title", "url", "content"}`` dicts (never raises)."""
    if settings.tavily_api_key:
        try:
            from langchain_tavily import TavilySearch

            raw = TavilySearch(max_results=max_results).invoke({"query": query})
            results = raw.get("results", []) if isinstance(raw, dict) else raw
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                for r in results or []
            ]
        except Exception as exc:
            log.warning("Tavily search failed (%s); falling back to DuckDuckGo.", exc)
    try:
        from ddgs import DDGS

        results = DDGS().text(query, region="us-en", max_results=max_results)
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
            for r in results or []
        ]
    except Exception as exc:
        log.warning("DuckDuckGo search failed: %s", exc)
        return []
