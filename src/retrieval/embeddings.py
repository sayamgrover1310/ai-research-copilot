"""Create local Ollama embeddings for document chunks and user queries."""

import math
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from src.config import get_embedding_model_name, get_ollama_base_url


def get_embedding_model() -> OllamaEmbeddings:
    """Create an Ollama embedding client using the configured local model."""
    return OllamaEmbeddings(
        model=get_embedding_model_name(),
        base_url=get_ollama_base_url(),
    )


def embed_chunks(chunks: list[Document], embedding_model: Embeddings) -> list[list[float]]:
    """Embed non-empty chunk text while leaving each Document and its metadata unchanged."""
    texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
    if not texts:
        return []

    return embedding_model.embed_documents(texts)


def embed_query(query: str, embedding_model: Embeddings) -> list[float]:
    """Embed one non-empty user query using the same model as document chunks."""
    if not query.strip():
        raise ValueError("A query must contain text before it can be embedded.")

    return embedding_model.embed_query(query)


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Return cosine similarity for equal-length, non-zero vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same number of dimensions.")

    magnitude_a = math.sqrt(sum(value * value for value in vector_a))
    magnitude_b = math.sqrt(sum(value * value for value in vector_b))
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    dot_product = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
    return dot_product / (magnitude_a * magnitude_b)
