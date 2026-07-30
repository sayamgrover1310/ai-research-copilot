"""Retrieve semantically relevant evidence chunks from a Chroma vector store."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrieval.embeddings import embed_query


DEFAULT_TOP_K = 4


def retrieve_documents(
    query: str,
    vector_store: Chroma,
    embedding_model: Embeddings,
    top_k: int = DEFAULT_TOP_K,
) -> list[Document]:
    """Return the most relevant stored chunks for a natural-language query."""
    results = retrieve_documents_with_scores(query, vector_store, embedding_model, top_k)
    return [document for document, _score in results]


def retrieve_documents_with_scores(
    query: str,
    vector_store: Chroma,
    embedding_model: Embeddings,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[Document, float]]:
    """Return relevant chunks with a higher-is-better similarity score for debugging."""
    _validate_top_k(top_k)
    query_vector = embed_query(query, embedding_model)

    # Chroma returns its distance value here: lower means a closer match. Convert it
    # to a simple similarity score so callers can read higher values as better.
    results_with_distances = vector_store.similarity_search_by_vector_with_relevance_scores(
        query_vector,
        k=top_k,
    )
    return [
        (document, 1 / (1 + distance))
        for document, distance in results_with_distances
    ]


def _validate_top_k(top_k: int) -> None:
    """Require a positive number of requested retrieval results."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
