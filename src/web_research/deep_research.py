"""Bounded planning, evidence collection, and synthesis for Deep Research."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.config import (
    get_deep_research_max_content_chars,
    get_deep_research_max_evidence,
    get_deep_research_max_queries,
    get_deep_research_results_per_query,
)
from src.rag.basic_rag import get_generation_model
from src.web_research.sources import WebSource
from src.web_research.tavily_search import search_web


MIN_RESEARCH_QUERIES = 2
NO_RESEARCH_EVIDENCE_ANSWER = "I could not find enough web evidence to complete this research request."


class ResearchPlan(BaseModel):
    """Structured planner output requested from the local generation model."""

    research_queries: list[str] = Field(
        description="Two to four focused web search queries for the research question."
    )


@dataclass(frozen=True)
class ResearchEvidence:
    """One deduplicated web source and the planned query that found it."""

    source: WebSource
    research_query: str


class ResearchPlanningError(Exception):
    """Raised when the planner cannot produce a valid bounded research plan."""


class ResearchSearchError(Exception):
    """Raised when planned searches cannot provide usable research evidence."""


class ResearchSynthesisError(Exception):
    """Raised when the local model cannot synthesize collected evidence."""


def plan_research(
    question: str,
    generation_model: Any | None = None,
    max_queries: int | None = None,
) -> list[str]:
    """Use structured output to turn one research question into 2-4 search queries."""
    if not question.strip():
        raise ValueError("A research question must contain text.")

    query_limit = max_queries if max_queries is not None else get_deep_research_max_queries()
    if not MIN_RESEARCH_QUERIES <= query_limit <= 4:
        raise ValueError("max_queries must be between 2 and 4.")

    model = generation_model or get_generation_model()
    try:
        structured_model = model.with_structured_output(ResearchPlan)
        plan = structured_model.invoke(build_research_planner_prompt(question, query_limit))
    except Exception as error:
        raise ResearchPlanningError(f"The research planner failed: {error}") from error

    if isinstance(plan, ResearchPlan):
        raw_queries = plan.research_queries
    elif isinstance(plan, dict):
        raw_queries = plan.get("research_queries", [])
    else:
        raise ResearchPlanningError("The research planner returned an unexpected structured result.")
    return normalize_research_queries(raw_queries, query_limit)


def normalize_research_queries(raw_queries: object, max_queries: int) -> list[str]:
    """Keep unique non-empty planner queries and reject plans outside the 2-4 bound."""
    if not isinstance(raw_queries, list):
        raise ResearchPlanningError("The research planner did not return a list of queries.")

    queries: list[str] = []
    for query in raw_queries:
        if not isinstance(query, str):
            continue
        cleaned_query = query.strip()
        if cleaned_query and cleaned_query not in queries:
            queries.append(cleaned_query)
        if len(queries) == max_queries:
            break

    if len(queries) < MIN_RESEARCH_QUERIES:
        raise ResearchPlanningError("The research planner must return at least two focused queries.")
    return queries


def collect_research_evidence(
    research_queries: list[str],
    search_function: Callable[[str, int | None], list[WebSource]] = search_web,
    results_per_query: int | None = None,
    max_evidence: int | None = None,
    max_content_chars: int | None = None,
) -> tuple[list[ResearchEvidence], list[WebSource]]:
    """Search each planned query once, then deduplicate evidence by provider URL."""
    result_limit = (
        results_per_query
        if results_per_query is not None
        else get_deep_research_results_per_query()
    )
    evidence_limit = max_evidence if max_evidence is not None else get_deep_research_max_evidence()
    content_limit = (
        max_content_chars
        if max_content_chars is not None
        else get_deep_research_max_content_chars()
    )

    if not research_queries:
        raise ResearchSearchError("The research plan did not contain any search queries.")

    evidence: list[ResearchEvidence] = []
    sources: list[WebSource] = []
    seen_urls: set[str] = set()
    search_failures: list[str] = []

    for research_query in research_queries:
        try:
            results = search_function(research_query, result_limit)
        except Exception as error:
            search_failures.append(str(error))
            continue

        for result in results:
            if result.url in seen_urls:
                continue

            seen_urls.add(result.url)
            normalized_source = WebSource(
                label=f"[W{len(sources) + 1}]",
                title=result.title,
                url=result.url,
                content=result.content[:content_limit],
                rank=result.rank,
            )
            sources.append(normalized_source)
            evidence.append(
                ResearchEvidence(
                    source=normalized_source,
                    research_query=research_query,
                )
            )

            if len(evidence) == evidence_limit:
                return evidence, sources

    if not evidence and search_failures:
        raise ResearchSearchError(f"All planned searches failed: {'; '.join(search_failures)}")
    return evidence, sources


def synthesize_research(
    question: str,
    evidence: list[ResearchEvidence],
    generation_model: Any | None = None,
) -> str:
    """Generate one answer from bounded, application-controlled web evidence."""
    if not evidence:
        raise ResearchSynthesisError("Research synthesis requires at least one evidence record.")

    model = generation_model or get_generation_model()
    try:
        response = model.invoke(build_research_synthesis_prompt(question, evidence))
        return _get_response_text(response)
    except Exception as error:
        raise ResearchSynthesisError(f"Research synthesis failed: {error}") from error


def build_research_planner_prompt(question: str, max_queries: int) -> str:
    """Ask the planner for a bounded structured search plan, not a prose response."""
    return f"""Create a focused web-research plan for the question below.

Return between {MIN_RESEARCH_QUERIES} and {max_queries} distinct search queries.
Each query should investigate a different useful aspect of the original question.
Do not answer the question yet.

Research question:
{question}
"""


def build_research_context(evidence: list[ResearchEvidence]) -> str:
    """Format collected evidence with its application-owned source metadata."""
    return "\n\n".join(
        f"{item.source.label} {item.source.title}\n"
        f"URL: {item.source.url}\n"
        f"Found for research query: {item.research_query}\n"
        f"Evidence: {item.source.content}"
        for item in evidence
    )


def build_research_synthesis_prompt(question: str, evidence: list[ResearchEvidence]) -> str:
    """Build a grounded synthesis prompt from the collected bounded evidence."""
    return f"""You are a careful research and study assistant.

Answer the original question using only the collected web evidence below.
Synthesize the evidence into a coherent explanation. If evidence is incomplete or uncertain,
say so clearly. Do not invent facts, titles, URLs, source details, or claims not supported by
the evidence.

Original research question:
{question}

Collected web evidence:
{build_research_context(evidence)}

Research synthesis:"""


def _get_response_text(response: Any) -> str:
    """Read text from a LangChain chat-model response."""
    content = getattr(response, "content", response)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The generation model returned an empty or non-text response.")
    return content.strip()
