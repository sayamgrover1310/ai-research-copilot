"""Tavily-specific web search kept separate from workflow and RAG logic."""

import json
from typing import Any

from src.config import get_tavily_api_key, get_web_search_max_results
from src.web_research.sources import WebSource


MAX_CONTENT_CHARS_PER_SOURCE = 1200


class WebSearchError(Exception):
    """Raised when the configured web-search provider cannot return results."""


def search_web(query: str, max_results: int | None = None) -> list[WebSource]:
    """Search Tavily and return normalized sources suitable for grounded answers."""
    if not query.strip():
        raise ValueError("A web search query must contain text.")

    result_limit = max_results if max_results is not None else get_web_search_max_results()
    if result_limit <= 0:
        raise ValueError("max_results must be a positive integer.")
    if not get_tavily_api_key():
        raise WebSearchError("TAVILY_API_KEY is required before web research can run.")

    try:
        from langchain_tavily import TavilySearch
    except ImportError as error:
        raise WebSearchError(
            "langchain-tavily is not installed. Install requirements.txt before web research."
        ) from error

    tool = TavilySearch(
        max_results=result_limit,
        search_depth="basic",
        include_answer=False,
        include_raw_content=True,
    )

    try:
        response = tool.invoke({"query": query})
    except Exception as error:
        raise WebSearchError(f"Tavily search failed: {error}") from error

    return normalize_tavily_results(response, result_limit)


def normalize_tavily_results(response: Any, max_results: int) -> list[WebSource]:
    """Keep valid Tavily results and safely ignore malformed provider responses."""
    response_data = _as_dictionary(response)
    raw_results = response_data.get("results", [])
    if not isinstance(raw_results, list):
        return []

    sources: list[WebSource] = []
    seen_urls: set[str] = set()

    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue

        url = _optional_text(raw_result.get("url"))
        content = _get_content(raw_result)
        if not url or not content or url in seen_urls:
            continue

        seen_urls.add(url)
        title = _optional_text(raw_result.get("title")) or url
        sources.append(
            WebSource(
                label=f"[W{len(sources) + 1}]",
                title=title,
                url=url,
                content=content[:MAX_CONTENT_CHARS_PER_SOURCE],
                rank=len(sources) + 1,
            )
        )
        if len(sources) == max_results:
            break

    return sources


def _as_dictionary(response: Any) -> dict[str, Any]:
    """Support Tavily's dictionary response and safely reject unexpected shapes."""
    if isinstance(response, dict):
        return response

    content = getattr(response, "content", response)
    if isinstance(content, str):
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return parsed_content if isinstance(parsed_content, dict) else {}
    return {}


def _get_content(result: dict[str, Any]) -> str | None:
    """Prefer cleaned page content, falling back to Tavily's search snippet."""
    return _optional_text(result.get("raw_content")) or _optional_text(result.get("content"))


def _optional_text(value: object) -> str | None:
    """Return a non-empty text value when Tavily supplied one."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
