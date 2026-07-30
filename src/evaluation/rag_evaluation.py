"""Evaluate retrieval and prepare human reviews of existing document RAG results.

This module deliberately does not use an LLM judge. Retrieval metrics are
deterministic, while answer groundedness and relevance remain human ratings for
the small evaluation sets appropriate to this project.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag.basic_rag import NO_CONTEXT_ANSWER, RAGResult, answer_question, build_context
from src.rag.citations import format_citation
from src.retrieval.retriever import retrieve_documents_with_scores


DEFAULT_K_VALUES = (1, 3, 5)


@dataclass(frozen=True)
class EvaluationExample:
    """One manually verified question and its expected document evidence."""

    question: str
    expected_source: str | None
    expected_page: int | None = None
    expected_chunk_id: str | None = None
    reference_answer: str | None = None
    expects_no_answer: bool = False
    groundedness_rating: int | None = None
    answer_relevance_rating: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class RetrievedEvidence:
    """Serializable information about one retrieved chunk used during evaluation."""

    rank: int
    score: float
    source_filename: str | None
    page_number: int | None
    chunk_id: str | None
    chunk_index: int | None
    text_preview: str
    is_relevant: bool


@dataclass(frozen=True)
class RetrievalQuestionResult:
    """Metric-relevant retrieval outcome for one evaluation example."""

    question: str
    expected_source: str | None
    expected_page: int | None
    expected_chunk_id: str | None
    first_relevant_rank: int | None
    retrieved_evidence: list[RetrievedEvidence]


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Aggregate retrieval metrics and transparent per-question outcomes."""

    question_count: int
    retrieval_question_count: int
    hit_rates: dict[int, float]
    mrr: float | None
    question_results: list[RetrievalQuestionResult]


@dataclass(frozen=True)
class GenerationReviewRecord:
    """Evidence and answer a human should inspect before assigning ratings."""

    question: str
    reference_answer: str | None
    generated_answer: str
    retrieved_context: str
    sources: list[str]
    expects_no_answer: bool
    abstained: bool
    error: str | None


def load_evaluation_dataset(dataset_path: str | Path) -> list[EvaluationExample]:
    """Load a small JSON list of manually written evaluation examples."""
    path = Path(dataset_path)
    try:
        raw_examples = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Evaluation dataset was not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Evaluation dataset must contain valid JSON: {error}") from error

    if not isinstance(raw_examples, list):
        raise ValueError("Evaluation dataset must be a JSON list of examples.")

    return [_example_from_mapping(example, index) for index, example in enumerate(raw_examples, start=1)]


def evaluate_retrieval(
    examples: list[EvaluationExample],
    vector_store: Chroma,
    embedding_model: Embeddings,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    retriever: Callable[[str, Chroma, Embeddings, int], list[tuple[Document, float]]] = retrieve_documents_with_scores,
) -> RetrievalEvaluationReport:
    """Measure whether expected local-document evidence appears in retrieved results.

    Questions marked ``expects_no_answer`` are retained in the per-question report,
    but excluded from Hit Rate and MRR because they intentionally have no relevant
    document chunk to retrieve.
    """
    normalized_k_values = _validate_k_values(k_values)
    max_k = max(normalized_k_values, default=0)
    question_results: list[RetrievalQuestionResult] = []

    for example in examples:
        retrieved = retriever(example.question, vector_store, embedding_model, max_k) if max_k else []
        evidence = [
            _to_retrieved_evidence(document, score, rank, example)
            for rank, (document, score) in enumerate(retrieved, start=1)
        ]
        first_relevant_rank = next(
            (item.rank for item in evidence if item.is_relevant),
            None,
        )
        question_results.append(
            RetrievalQuestionResult(
                question=example.question,
                expected_source=example.expected_source,
                expected_page=example.expected_page,
                expected_chunk_id=example.expected_chunk_id,
                first_relevant_rank=first_relevant_rank,
                retrieved_evidence=evidence,
            )
        )

    retrieval_results = [
        result
        for example, result in zip(examples, question_results)
        if not example.expects_no_answer
    ]
    hit_rates = {
        k: _mean(result.first_relevant_rank is not None and result.first_relevant_rank <= k for result in retrieval_results)
        for k in normalized_k_values
    }
    mrr = _mean(
        1 / result.first_relevant_rank if result.first_relevant_rank is not None else 0
        for result in retrieval_results
    ) if retrieval_results else None

    return RetrievalEvaluationReport(
        question_count=len(examples),
        retrieval_question_count=len(retrieval_results),
        hit_rates=hit_rates,
        mrr=mrr,
        question_results=question_results,
    )


def build_generation_review_records(
    examples: list[EvaluationExample],
    vector_store: Chroma,
    embedding_model: Embeddings,
    generation_model: Any | None = None,
    top_k: int = 4,
    rag_runner: Callable[..., RAGResult] = answer_question,
) -> list[GenerationReviewRecord]:
    """Run existing RAG and return evidence for deterministic human review.

    A reviewer can add 0-2 ratings back to the JSON dataset: groundedness is
    unsupported/partially supported/fully supported; relevance is irrelevant/
    partially relevant/directly relevant.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    records: list[GenerationReviewRecord] = []
    for example in examples:
        result = rag_runner(
            example.question,
            vector_store,
            embedding_model,
            generation_model=generation_model,
            top_k=top_k,
        )
        records.append(
            GenerationReviewRecord(
                question=example.question,
                reference_answer=example.reference_answer,
                generated_answer=result.answer,
                retrieved_context=build_context(result.source_documents),
                sources=[format_citation(source) for source in result.sources],
                expects_no_answer=example.expects_no_answer,
                abstained=result.answer == NO_CONTEXT_ANSWER,
                error=result.error,
            )
        )
    return records


def calculate_human_rating_averages(examples: list[EvaluationExample]) -> dict[str, float | None]:
    """Average optional 0-2 human ratings already stored in the dataset."""
    groundedness = [example.groundedness_rating for example in examples if example.groundedness_rating is not None]
    relevance = [example.answer_relevance_rating for example in examples if example.answer_relevance_rating is not None]
    return {
        "groundedness": _mean(groundedness) if groundedness else None,
        "answer_relevance": _mean(relevance) if relevance else None,
    }


def render_retrieval_report(report: RetrievalEvaluationReport) -> str:
    """Create a compact terminal report with aggregate and failed-case details."""
    lines = [
        "Document RAG Retrieval Evaluation",
        f"Evaluation questions: {report.question_count}",
        f"Questions with expected evidence: {report.retrieval_question_count}",
    ]
    lines.extend(f"Hit Rate@{k}: {score:.2%}" for k, score in report.hit_rates.items())
    lines.append(f"MRR: {report.mrr:.3f}" if report.mrr is not None else "MRR: not applicable")

    failed = [
        result
        for result in report.question_results
        if result.expected_source is not None and result.first_relevant_rank is None
    ]
    lines.append(f"Failed retrieval cases: {len(failed)}")
    for result in failed:
        page = f", page {result.expected_page}" if result.expected_page is not None else ""
        lines.append(f"- {result.question!r} (expected {result.expected_source}{page})")

    lines.append("Per-question retrieval results:")
    for result in report.question_results:
        rank = str(result.first_relevant_rank) if result.first_relevant_rank is not None else "not found"
        lines.append(f"- {result.question!r}: first relevant rank = {rank}")
        for evidence in result.retrieved_evidence:
            page = f", page {evidence.page_number}" if evidence.page_number is not None else ""
            marker = "relevant" if evidence.is_relevant else "not relevant"
            lines.append(
                f"  [{evidence.rank}] {evidence.source_filename or 'Unknown source'}{page} "
                f"score={evidence.score:.3f} ({marker})"
            )
    return "\n".join(lines)


def save_evaluation_results(
    output_path: str | Path,
    retrieval_report: RetrievalEvaluationReport,
    generation_records: list[GenerationReviewRecord] | None = None,
    human_rating_averages: dict[str, float | None] | None = None,
) -> None:
    """Save machine-readable results for later comparison without a dashboard."""
    payload: dict[str, object] = {"retrieval": asdict(retrieval_report)}
    if generation_records is not None:
        payload["generation_review"] = [asdict(record) for record in generation_records]
    if human_rating_averages is not None:
        payload["human_rating_averages"] = human_rating_averages
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _example_from_mapping(raw_example: object, index: int) -> EvaluationExample:
    """Validate one editable JSON example with clear errors for beginners."""
    if not isinstance(raw_example, dict):
        raise ValueError(f"Evaluation example {index} must be a JSON object.")

    question = _optional_text(raw_example.get("question"))
    if question is None:
        raise ValueError(f"Evaluation example {index} must include a non-empty question.")

    expects_no_answer = raw_example.get("expects_no_answer", False)
    if not isinstance(expects_no_answer, bool):
        raise ValueError(f"Evaluation example {index} expects_no_answer must be true or false.")

    expected_source = _optional_text(raw_example.get("expected_source"))
    expected_page = _optional_positive_integer(raw_example.get("expected_page"), "expected_page", index)
    expected_chunk_id = _optional_text(raw_example.get("expected_chunk_id"))
    if expects_no_answer and any(value is not None for value in (expected_source, expected_page, expected_chunk_id)):
        raise ValueError(f"Evaluation example {index} cannot expect a source and no answer together.")
    if not expects_no_answer and expected_source is None:
        raise ValueError(f"Evaluation example {index} must include expected_source.")

    return EvaluationExample(
        question=question,
        expected_source=expected_source,
        expected_page=expected_page,
        expected_chunk_id=expected_chunk_id,
        reference_answer=_optional_text(raw_example.get("reference_answer")),
        expects_no_answer=expects_no_answer,
        groundedness_rating=_optional_rating(raw_example.get("groundedness_rating"), "groundedness_rating", index),
        answer_relevance_rating=_optional_rating(raw_example.get("answer_relevance_rating"), "answer_relevance_rating", index),
        notes=_optional_text(raw_example.get("notes")),
    )


def _to_retrieved_evidence(
    document: Document,
    score: float,
    rank: int,
    example: EvaluationExample,
) -> RetrievedEvidence:
    metadata = document.metadata
    return RetrievedEvidence(
        rank=rank,
        score=score,
        source_filename=_document_source_filename(document),
        page_number=metadata.get("page_number") if isinstance(metadata.get("page_number"), int) else None,
        chunk_id=metadata.get("chunk_id") if isinstance(metadata.get("chunk_id"), str) else None,
        chunk_index=metadata.get("chunk_index") if isinstance(metadata.get("chunk_index"), int) else None,
        text_preview=document.page_content.replace("\n", " ")[:160],
        is_relevant=_is_relevant_document(document, example),
    )


def _is_relevant_document(document: Document, example: EvaluationExample) -> bool:
    """Match only metadata supplied by the human who wrote the evaluation example."""
    if example.expects_no_answer:
        return False

    metadata = document.metadata
    if _document_source_filename(document) != example.expected_source:
        return False
    if example.expected_page is not None and metadata.get("page_number") != example.expected_page:
        return False
    return example.expected_chunk_id is None or metadata.get("chunk_id") == example.expected_chunk_id


def _document_source_filename(document: Document) -> str | None:
    filename = document.metadata.get("source_filename")
    if isinstance(filename, str) and filename.strip():
        return filename
    source = document.metadata.get("source")
    return Path(source).name if isinstance(source, str) and source.strip() else None


def _validate_k_values(k_values: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(k_values)
    if not normalized:
        raise ValueError("k_values must contain at least one positive integer.")
    if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in normalized):
        raise ValueError("Each K value must be a positive integer.")
    return tuple(sorted(set(normalized)))


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_positive_integer(value: object, field_name: str, index: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Evaluation example {index} {field_name} must be a positive integer when provided.")
    return value


def _optional_rating(value: object, field_name: str, index: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2):
        raise ValueError(f"Evaluation example {index} {field_name} must be 0, 1, or 2 when provided.")
    return value


def _mean(values: Iterable[float | bool | int]) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0
