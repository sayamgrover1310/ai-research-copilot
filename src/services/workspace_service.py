"""Workspace-aware document processing for interfaces such as Streamlit."""

from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma

from src.config import get_uploads_directory
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_document
from src.retrieval.vector_store import get_vector_store


@dataclass(frozen=True)
class UploadedDocumentData:
    """File data supplied by an interface without depending on that interface's types."""

    filename: str
    content: bytes


@dataclass
class DocumentProcessingResult:
    """A concise report of one explicit document-indexing action."""

    processed_files: list[str]
    indexed_chunk_ids: list[str]
    errors: list[str]

    @property
    def chunks_indexed(self) -> int:
        """Return the number of non-empty chunks accepted by persistent storage."""
        return len(self.indexed_chunk_ids)


def process_uploaded_documents(
    uploaded_documents: list[UploadedDocumentData],
    vector_store: Chroma,
    workspace_id: str,
    uploads_directory: str | Path | None = None,
) -> DocumentProcessingResult:
    """Save uploads, then reuse existing ingestion, chunking, and Chroma storage.

    Processing happens only when an interface explicitly calls this function. Stable
    chunk IDs make reprocessing unchanged files safe.
    """
    workspace_directory = Path(uploads_directory or get_uploads_directory()) / workspace_id
    workspace_directory.mkdir(parents=True, exist_ok=True)

    processed_files: list[str] = []
    indexed_chunk_ids: list[str] = []
    errors: list[str] = []

    for upload in uploaded_documents:
        filename = Path(upload.filename).name
        if not filename:
            errors.append("An uploaded file did not have a valid filename.")
            continue

        file_path = workspace_directory / filename
        try:
            file_path.write_bytes(upload.content)
            documents = load_document(file_path)
            chunks = chunk_documents(documents)
            chunk_ids = get_vector_store_chunk_ids(chunks, vector_store)
        except Exception as error:
            errors.append(f"{filename}: {error}")
            continue

        processed_files.append(filename)
        indexed_chunk_ids.extend(chunk_ids)

    return DocumentProcessingResult(processed_files, indexed_chunk_ids, errors)


def get_workspace_document_count(vector_store: Chroma) -> int:
    """Return the number of indexed chunks in the currently selected workspace."""
    return len(vector_store.get().get("ids", []))


def get_vector_store_chunk_ids(chunks: list, vector_store: Chroma) -> list[str]:
    """Keep the storage call separately named for readable UI-service flow."""
    from src.retrieval.vector_store import store_chunks

    return store_chunks(chunks, vector_store)
