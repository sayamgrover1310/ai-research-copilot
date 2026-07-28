"""Persist document chunks and embeddings in local Chroma collections."""

import hashlib
import json
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import get_chroma_persist_directory
from src.retrieval.embeddings import get_embedding_model


WORKSPACE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,40}")


def get_vector_store(
    embedding_model: Embeddings | None = None,
    workspace_id: str = "default",
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Open the persistent Chroma collection for one research workspace."""
    _validate_workspace_id(workspace_id)

    directory = Path(persist_directory) if persist_directory else get_chroma_persist_directory()
    directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=f"research_workspace_{workspace_id}",
        embedding_function=embedding_model or get_embedding_model(),
        persist_directory=str(directory),
    )


def store_chunks(chunks: list[Document], vector_store: Chroma) -> list[str]:
    """Store unique, non-empty chunks and return their stable Chroma IDs.

    Existing IDs are skipped so running ingestion again with unchanged chunks is safe.
    """
    documents_to_store: list[Document] = []
    chunk_ids: list[str] = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        if not chunk.page_content.strip():
            continue

        chunk_id = create_chunk_id(chunk)
        if chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)
        chunk_ids.append(chunk_id)
        documents_to_store.append(_document_with_chunk_id(chunk, chunk_id))

    if not documents_to_store:
        return []

    existing_ids = set(vector_store.get(ids=chunk_ids)["ids"])
    new_documents = [
        document
        for document, chunk_id in zip(documents_to_store, chunk_ids)
        if chunk_id not in existing_ids
    ]
    new_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in existing_ids]

    if new_documents:
        vector_store.add_documents(documents=new_documents, ids=new_ids)

    return chunk_ids


def create_chunk_id(chunk: Document) -> str:
    """Create a stable ID from chunk identity metadata and exact text content."""
    _validate_chunk_metadata(chunk)

    metadata = chunk.metadata
    identity = {
        "source": metadata.get("source", metadata["source_filename"]),
        "source_filename": metadata["source_filename"],
        "page_number": metadata.get("page_number"),
        "chunk_index": metadata["chunk_index"],
        "page_content": chunk.page_content,
    }
    encoded_identity = json.dumps(identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return f"chunk_{hashlib.sha256(encoded_identity).hexdigest()}"


def _document_with_chunk_id(chunk: Document, chunk_id: str) -> Document:
    """Copy a chunk and add its stable ID to metadata stored in Chroma."""
    metadata = dict(chunk.metadata)
    metadata["chunk_id"] = chunk_id
    return Document(page_content=chunk.page_content, metadata=metadata)


def _validate_workspace_id(workspace_id: str) -> None:
    """Keep future workspace collection names valid and predictable."""
    if not WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
        raise ValueError(
            "workspace_id must contain 1-40 letters, numbers, underscores, or hyphens."
        )


def _validate_chunk_metadata(chunk: Document) -> None:
    """Ensure metadata can be safely persisted by Chroma and used for stable IDs."""
    metadata = chunk.metadata

    if not isinstance(metadata.get("source_filename"), str) or not metadata["source_filename"]:
        raise ValueError("Chunk metadata must include a non-empty source_filename.")

    chunk_index = metadata.get("chunk_index")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("Chunk metadata must include a non-negative integer chunk_index.")

    if "page_number" in metadata:
        page_number = metadata["page_number"]
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
            raise ValueError("page_number must be a positive integer when provided.")

    for key, value in metadata.items():
        if not isinstance(value, str | int | float | bool):
            raise ValueError(f"Metadata value for '{key}' is not supported by Chroma.")
