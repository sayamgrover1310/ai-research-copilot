"""Configuration values shared by application components."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHROMA_PERSIST_DIRECTORY = "data/chroma"


def get_ollama_base_url() -> str:
    """Return the local Ollama server URL."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def get_embedding_model_name() -> str:
    """Return the configured local embedding model name."""
    return os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_EMBEDDING_MODEL)


def get_chroma_persist_directory() -> Path:
    """Return the local directory where Chroma persists vector data."""
    directory = os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_PERSIST_DIRECTORY)
    return Path(directory)
