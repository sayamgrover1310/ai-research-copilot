"""Generate grounded answers from one round of web-search evidence."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.config import get_web_search_max_results
from src.rag.basic_rag import get_generation_model
from src.web_research.sources import WebSource
from src.web_research.tavily_search import WebSearchError, search_web


NO_WEB_EVIDENCE_ANSWER = "I could not find enough web evidence to answer that question."


@dataclass
class WebResearchResult:
    """A web-grounded answer and the provider-returned sources used as evidence."""

    answer: str
    web_sources: list[WebSource]
    error: str | None = None


def answer_web_question(
    question: str,
    generation_model: Any | None = None,
    max_results: int | None = None,
    search_function: Callable[[str, int | None], list[WebSource]] = search_web,
) -> WebResearchResult:
    """Search once, then ask the existing Ollama model to answer from web evidence."""
    if not question.strip():
        raise ValueError("A question must contain text before web research can run.")

    result_limit = max_results if max_results is not None else get_web_search_max_results()
    try:
        web_sources = search_function(question, result_limit)
    except WebSearchError as error:
        return WebResearchResult(answer="", web_sources=[], error=str(error))
    except Exception as error:
        return WebResearchResult(
            answer="",
            web_sources=[],
            error=f"Web search could not return results: {error}",
        )

    if not web_sources:
        return WebResearchResult(answer=NO_WEB_EVIDENCE_ANSWER, web_sources=[])

    prompt = build_web_research_prompt(question, web_sources)
    model = generation_model or get_generation_model()

    try:
        response = model.invoke(prompt)
        answer = _get_response_text(response)
    except Exception as error:
        return WebResearchResult(
            answer="",
            web_sources=web_sources,
            error=f"The local generation model could not produce a web-grounded answer: {error}",
        )

    return WebResearchResult(answer=answer, web_sources=web_sources)


def build_web_context(web_sources: list[WebSource]) -> str:
    """Format provider-returned evidence without changing its source metadata."""
    return "\n\n".join(
        f"{source.label} {source.title}\nURL: {source.url}\nEvidence: {source.content}"
        for source in web_sources
    )


def build_web_research_prompt(question: str, web_sources: list[WebSource]) -> str:
    """Build a prompt that limits answer generation to retrieved web evidence."""
    return f"""You are a helpful research and study assistant.

Answer the user's question using only the web evidence below.
If the evidence is insufficient, say that you do not have enough evidence.
Do not invent facts, titles, URLs, or source details.

User question:
{question}

Web evidence:
{build_web_context(web_sources)}

Answer:"""


def _get_response_text(response: Any) -> str:
    """Read plain text from a LangChain chat-model response."""
    content = getattr(response, "content", response)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The generation model returned an empty or non-text response.")
    return content.strip()
