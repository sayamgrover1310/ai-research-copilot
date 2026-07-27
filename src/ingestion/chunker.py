"""Split ingested LangChain Documents into smaller, overlapping chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split non-empty Documents and preserve their metadata.

    The default settings can be overridden when comparing chunking strategies later.
    """
    _validate_chunk_settings(chunk_size, chunk_overlap)

    non_empty_documents = [
        document for document in documents if document.page_content.strip()
    ]
    if not non_empty_documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_documents = text_splitter.split_documents(non_empty_documents)

    return [
        _add_chunk_metadata(document, chunk_index)
        for chunk_index, document in enumerate(split_documents)
    ]


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Ensure the text splitter receives sensible settings."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def _add_chunk_metadata(document: Document, chunk_index: int) -> Document:
    """Return a chunk with all original metadata plus its position in the result."""
    metadata = dict(document.metadata)
    metadata["chunk_index"] = chunk_index

    return Document(page_content=document.page_content, metadata=metadata)
