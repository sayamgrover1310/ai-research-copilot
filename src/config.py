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


def get_chroma_persist_directory() -> Path:
    """Return the local directory where Chroma persists vector data."""
    directory = os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_PERSIST_DIRECTORY)
    return Path(directory)
