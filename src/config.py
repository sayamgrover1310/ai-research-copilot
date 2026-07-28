"""Configuration values shared by application components."""

import os

from dotenv import load_dotenv


load_dotenv()

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


def get_ollama_base_url() -> str:
    """Return the local Ollama server URL."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def get_embedding_model_name() -> str:
    """Return the configured local embedding model name."""
    return os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_EMBEDDING_MODEL)
