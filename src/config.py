"""Configuration values shared by application components."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_GENERATION_MODEL = "qwen3:4b-instruct"
DEFAULT_CHROMA_PERSIST_DIRECTORY = "data/chroma"
DEFAULT_WEB_SEARCH_MAX_RESULTS = 3
DEFAULT_DEEP_RESEARCH_MAX_QUERIES = 4
DEFAULT_DEEP_RESEARCH_RESULTS_PER_QUERY = 2
DEFAULT_DEEP_RESEARCH_MAX_EVIDENCE = 6
DEFAULT_DEEP_RESEARCH_MAX_CONTENT_CHARS = 600
DEFAULT_STUDY_QUIZ_MAX_QUESTIONS = 10
DEFAULT_STUDY_FLASHCARD_MAX_COUNT = 20


def get_ollama_base_url() -> str:
    """Return the local Ollama server URL."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def get_embedding_model_name() -> str:
    """Return the configured local embedding model name."""
    return os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_EMBEDDING_MODEL)


def get_generation_model_name() -> str:
    """Return the configured local model used to generate RAG answers."""
    return os.getenv("OLLAMA_GENERATION_MODEL", DEFAULT_OLLAMA_GENERATION_MODEL)


def get_tavily_api_key() -> str | None:
    """Return the optional API key used by the Tavily web-search provider."""
    return os.getenv("TAVILY_API_KEY")


def get_web_search_max_results() -> int:
    """Return a positive, configurable limit for web-search results."""
    value = os.getenv("WEB_SEARCH_MAX_RESULTS", str(DEFAULT_WEB_SEARCH_MAX_RESULTS))
    try:
        max_results = int(value)
    except ValueError as error:
        raise ValueError("WEB_SEARCH_MAX_RESULTS must be a positive integer.") from error

    if max_results <= 0:
        raise ValueError("WEB_SEARCH_MAX_RESULTS must be a positive integer.")
    return max_results


def get_deep_research_max_queries() -> int:
    """Return the bounded number of planner queries for one research request."""
    value = _get_positive_integer("DEEP_RESEARCH_MAX_QUERIES", DEFAULT_DEEP_RESEARCH_MAX_QUERIES)
    if not 2 <= value <= 4:
        raise ValueError("DEEP_RESEARCH_MAX_QUERIES must be between 2 and 4.")
    return value


def get_deep_research_results_per_query() -> int:
    """Return the bounded number of provider results per planned query."""
    return _get_positive_integer(
        "DEEP_RESEARCH_RESULTS_PER_QUERY",
        DEFAULT_DEEP_RESEARCH_RESULTS_PER_QUERY,
    )


def get_deep_research_max_evidence() -> int:
    """Return the maximum unique evidence records passed to synthesis."""
    return _get_positive_integer("DEEP_RESEARCH_MAX_EVIDENCE", DEFAULT_DEEP_RESEARCH_MAX_EVIDENCE)


def get_deep_research_max_content_chars() -> int:
    """Return the evidence-content limit used to bound synthesis context."""
    return _get_positive_integer(
        "DEEP_RESEARCH_MAX_CONTENT_CHARS",
        DEFAULT_DEEP_RESEARCH_MAX_CONTENT_CHARS,
    )


def get_study_quiz_max_questions() -> int:
    """Return the maximum bounded number of quiz questions per request."""
    return _get_positive_integer("STUDY_QUIZ_MAX_QUESTIONS", DEFAULT_STUDY_QUIZ_MAX_QUESTIONS)


def get_study_flashcard_max_count() -> int:
    """Return the maximum bounded number of flashcards per request."""
    return _get_positive_integer("STUDY_FLASHCARD_MAX_COUNT", DEFAULT_STUDY_FLASHCARD_MAX_COUNT)


def get_chroma_persist_directory() -> Path:
    """Return the local directory where Chroma persists vector data."""
    directory = os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_PERSIST_DIRECTORY)
    return Path(directory)


def _get_positive_integer(variable_name: str, default: int) -> int:
    """Read one positive integer environment setting with a clear error message."""
    value = os.getenv(variable_name, str(default))
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise ValueError(f"{variable_name} must be a positive integer.") from error

    if parsed_value <= 0:
        raise ValueError(f"{variable_name} must be a positive integer.")
    return parsed_value
