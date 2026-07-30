"""Build deterministic local-document source records from retrieved Documents."""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


@dataclass(frozen=True)
class CitationSource:
    """One displayable local-document source, derived only from Document metadata."""

    label: str
    source: str | None
    source_filename: str | None
    page_number: int | None
    chunk_indices: tuple[int, ...]
    chunk_ids: tuple[str, ...]


def build_citation_sources(documents: list[Document]) -> list[CitationSource]:
    """Create ordered, deduplicated source records from retrieved Documents.

    Chunks from the same file and page share one citation label. Their distinct chunk
    indexes and IDs are retained inside that source record for later citation work.
    """
    grouped_sources: dict[tuple[str, int | None], dict[str, object]] = {}

    for document in documents:
        metadata = document.metadata
        source = _optional_text(metadata.get("source"))
        source_filename = _optional_text(metadata.get("source_filename"))
        page_number = _optional_page_number(metadata.get("page_number"))

        # Prefer the complete source path for identity. Fall back to filename when a
        # source path was not available from the loader.
        source_identity = source or source_filename or "unknown-source"
        key = (source_identity, page_number)

        if key not in grouped_sources:
            grouped_sources[key] = {
                "source": source,
                "source_filename": source_filename,
                "page_number": page_number,
                "chunk_indices": [],
                "chunk_ids": [],
            }

        record = grouped_sources[key]
        _append_unique(record["chunk_indices"], _optional_non_negative_int(metadata.get("chunk_index")))
        _append_unique(record["chunk_ids"], _optional_text(metadata.get("chunk_id")))

    return [
        CitationSource(
            label=f"[{index}]",
            source=record["source"],
            source_filename=record["source_filename"],
            page_number=record["page_number"],
            chunk_indices=tuple(record["chunk_indices"]),
            chunk_ids=tuple(record["chunk_ids"]),
        )
        for index, record in enumerate(grouped_sources.values(), start=1)
    ]


def format_citation(source: CitationSource) -> str:
    """Return a short, user-facing source line without inventing missing details."""
    filename = source.source_filename or _filename_from_path(source.source) or "Unknown source"
    page = f" — Page {source.page_number}" if source.page_number is not None else ""
    return f"{source.label} {filename}{page}"


def _optional_text(value: object) -> str | None:
    """Return a non-empty string metadata value, if one is available."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _optional_page_number(value: object) -> int | None:
    """Accept only positive, one-based page numbers from existing metadata."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return None


def _optional_non_negative_int(value: object) -> int | None:
    """Accept only non-negative chunk indexes from existing metadata."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _append_unique(values: object, value: str | int | None) -> None:
    """Append metadata values once while retaining retrieval order."""
    if value is not None and value not in values:
        values.append(value)


def _filename_from_path(source: str | None) -> str | None:
    """Use a path's filename only as a display fallback."""
    return Path(source).name if source else None
