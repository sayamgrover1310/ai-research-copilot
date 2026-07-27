"""Load supported local files into LangChain Document objects."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


SUPPORTED_FILE_TYPES = {".pdf", ".txt"}


class DocumentIngestionError(Exception):
    """Raised when a supported file cannot be read as text."""


def load_document(file_path: str | Path) -> list[Document]:
    """Load one PDF or TXT file and return non-empty LangChain Documents.

    PDF files produce one Document per page when text can be extracted. TXT files
    produce one Document for the complete file. An empty document returns an empty list.
    """
    path = Path(file_path)
    _validate_file(path)

    try:
        documents = _choose_loader(path).load()
    except Exception as error:
        raise DocumentIngestionError(
            f"Could not extract text from '{path.name}'. "
            "Make sure the file is readable and contains extractable text."
        ) from error

    non_empty_documents = [
        document for document in documents if document.page_content.strip()
    ]

    return [_add_metadata(document, path) for document in non_empty_documents]


def _validate_file(path: Path) -> None:
    """Check that the input points to an existing supported file."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FILE_TYPES:
        supported_types = ", ".join(sorted(SUPPORTED_FILE_TYPES))
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported types: {supported_types}."
        )


def _choose_loader(path: Path) -> PyPDFLoader | TextLoader:
    """Return the LangChain loader that matches the file extension."""
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path))

    return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)


def _add_metadata(document: Document, path: Path) -> Document:
    """Add consistent metadata used by later retrieval and citations."""
    metadata = dict(document.metadata)
    metadata["source"] = str(path)
    metadata["source_filename"] = path.name
    metadata["file_type"] = path.suffix.lower().lstrip(".")

    # PyPDFLoader uses a zero-based "page" value. page_number is friendlier for users.
    if "page" in metadata:
        metadata["page_number"] = int(metadata["page"]) + 1

    return Document(page_content=document.page_content, metadata=metadata)
