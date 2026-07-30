"""Create grounded summaries, quizzes, and flashcards from uploaded documents."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from src.config import get_study_flashcard_max_count, get_study_quiz_max_questions
from src.rag.basic_rag import build_context, get_generation_model
from src.rag.citations import CitationSource, build_citation_sources
from src.retrieval.retriever import DEFAULT_TOP_K, retrieve_documents


NO_STUDY_EVIDENCE_MESSAGE = "I could not find relevant material in the uploaded documents for this topic."


class QuizQuestion(BaseModel):
    """One multiple-choice question generated from retrieved study evidence."""

    question: str
    options: list[str]
    correct_answer: str
    explanation: str


class QuizOutput(BaseModel):
    """Structured model output for a requested set of quiz questions."""

    questions: list[QuizQuestion] = Field(description="Grounded multiple-choice quiz questions.")


class Flashcard(BaseModel):
    """One concise question-and-answer study card."""

    front: str
    back: str


class FlashcardOutput(BaseModel):
    """Structured model output for a requested set of flashcards."""

    flashcards: list[Flashcard] = Field(description="Grounded concise flashcards.")


@dataclass
class TopicSummaryResult:
    """A grounded topic summary and its retrieved evidence."""

    summary: str
    source_documents: list[Document]
    sources: list[CitationSource]
    error: str | None = None


@dataclass
class QuizResult:
    """A generated quiz and the retrieved evidence used to create it."""

    questions: list[QuizQuestion]
    source_documents: list[Document]
    sources: list[CitationSource]
    error: str | None = None


@dataclass
class FlashcardResult:
    """Generated flashcards and the retrieved evidence used to create them."""

    flashcards: list[Flashcard]
    source_documents: list[Document]
    sources: list[CitationSource]
    error: str | None = None


class StudyMode:
    """Reusable service that generates study materials from existing document retrieval."""

    def __init__(
        self,
        vector_store: Chroma,
        embedding_model: Embeddings,
        generation_model: Any | None = None,
        retriever: Callable[[str, Chroma, Embeddings, int], list[Document]] = retrieve_documents,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.retriever = retriever
        self.top_k = top_k

    def summarize(self, topic: str) -> TopicSummaryResult:
        """Retrieve local evidence and generate a grounded study summary for one topic."""
        documents, sources, error = self._retrieve_topic(topic)
        if error:
            return TopicSummaryResult("", documents, sources, error)
        if not documents:
            return TopicSummaryResult(NO_STUDY_EVIDENCE_MESSAGE, [], [])

        try:
            response = self._model().invoke(build_summary_prompt(topic, documents))
            summary = _get_response_text(response)
        except Exception as error:
            return TopicSummaryResult("", documents, sources, f"Study summary generation failed: {error}")

        return TopicSummaryResult(summary, documents, sources)

    def generate_quiz(self, topic: str, count: int) -> QuizResult:
        """Retrieve local evidence and generate a validated multiple-choice quiz."""
        try:
            _validate_count(count, "count", get_study_quiz_max_questions())
        except ValueError as error:
            return QuizResult([], [], [], str(error))

        documents, sources, error = self._retrieve_topic(topic)
        if error:
            return QuizResult([], documents, sources, error)
        if not documents:
            return QuizResult([], [], [], NO_STUDY_EVIDENCE_MESSAGE)

        try:
            structured_model = self._model().with_structured_output(QuizOutput)
            output = structured_model.invoke(build_quiz_prompt(topic, count, documents))
            questions = _read_quiz_output(output, count)
        except Exception as error:
            return QuizResult([], documents, sources, f"Quiz generation failed: {error}")

        return QuizResult(questions, documents, sources)

    def generate_flashcards(self, topic: str, count: int) -> FlashcardResult:
        """Retrieve local evidence and generate validated concise flashcards."""
        try:
            _validate_count(count, "count", get_study_flashcard_max_count())
        except ValueError as error:
            return FlashcardResult([], [], [], str(error))

        documents, sources, error = self._retrieve_topic(topic)
        if error:
            return FlashcardResult([], documents, sources, error)
        if not documents:
            return FlashcardResult([], [], [], NO_STUDY_EVIDENCE_MESSAGE)

        try:
            structured_model = self._model().with_structured_output(FlashcardOutput)
            output = structured_model.invoke(build_flashcard_prompt(topic, count, documents))
            flashcards = _read_flashcard_output(output, count)
        except Exception as error:
            return FlashcardResult([], documents, sources, f"Flashcard generation failed: {error}")

        return FlashcardResult(flashcards, documents, sources)

    def _retrieve_topic(
        self,
        topic: str,
    ) -> tuple[list[Document], list[CitationSource], str | None]:
        """Use the existing semantic retriever and existing citation metadata flow."""
        if not topic.strip():
            return [], [], "A study topic must contain text."

        try:
            documents = self.retriever(topic, self.vector_store, self.embedding_model, self.top_k)
        except Exception as error:
            return [], [], f"Study retrieval failed: {error}"

        return documents, build_citation_sources(documents), None

    def _model(self) -> Any:
        """Use an injected test model or the existing configured Ollama generation model."""
        return self.generation_model or get_generation_model()


def build_summary_prompt(topic: str, documents: list[Document]) -> str:
    """Build a summary prompt grounded solely in retrieved document chunks."""
    return f"""You are a helpful study assistant.

Create a clear, concise summary of the topic using only the retrieved study material.
If the material is insufficient, say so. Do not invent information.

Topic:
{topic}

Retrieved study material:
{build_context(documents)}

Summary:"""


def build_quiz_prompt(topic: str, count: int, documents: list[Document]) -> str:
    """Build a structured-output prompt for grounded four-option MCQs."""
    return f"""You are a helpful study assistant.

Create exactly {count} multiple-choice questions about the topic using only the retrieved
study material. Each question must have exactly four distinct options, one correct answer that
matches one option exactly, and a short explanation. Do not invent information.

Topic:
{topic}

Retrieved study material:
{build_context(documents)}
"""


def build_flashcard_prompt(topic: str, count: int, documents: list[Document]) -> str:
    """Build a structured-output prompt for concise grounded flashcards."""
    return f"""You are a helpful study assistant.

Create exactly {count} concise flashcards about the topic using only the retrieved study material.
Each flashcard needs a clear question or term on the front and a concise answer on the back.
Do not invent information.

Topic:
{topic}

Retrieved study material:
{build_context(documents)}
"""


def _read_quiz_output(output: object, expected_count: int) -> list[QuizQuestion]:
    """Validate structured quiz output before returning it to the caller."""
    parsed_output = output if isinstance(output, QuizOutput) else QuizOutput.model_validate(output)
    if len(parsed_output.questions) != expected_count:
        raise ValueError(f"The model returned {len(parsed_output.questions)} questions; expected {expected_count}.")

    for question in parsed_output.questions:
        options = [option.strip() for option in question.options if option.strip()]
        if len(options) != 4 or len(set(options)) != 4:
            raise ValueError("Each quiz question must contain four distinct non-empty options.")
        if question.correct_answer not in options:
            raise ValueError("Each quiz question's correct answer must match one option exactly.")
        if not question.question.strip() or not question.explanation.strip():
            raise ValueError("Each quiz question needs text and a short explanation.")
        question.options = options

    return parsed_output.questions


def _read_flashcard_output(output: object, expected_count: int) -> list[Flashcard]:
    """Validate structured flashcard output before returning it to the caller."""
    parsed_output = output if isinstance(output, FlashcardOutput) else FlashcardOutput.model_validate(output)
    if len(parsed_output.flashcards) != expected_count:
        raise ValueError(f"The model returned {len(parsed_output.flashcards)} flashcards; expected {expected_count}.")

    for flashcard in parsed_output.flashcards:
        if not flashcard.front.strip() or not flashcard.back.strip():
            raise ValueError("Each flashcard needs non-empty front and back text.")
    return parsed_output.flashcards


def _validate_count(count: int, name: str, maximum: int) -> None:
    """Keep generation requests positive and bounded for predictable local runtime."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    if count > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")


def _get_response_text(response: Any) -> str:
    """Read non-empty plain text from the existing LangChain chat-model response."""
    content = getattr(response, "content", response)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The generation model returned an empty or non-text response.")
    return content.strip()
