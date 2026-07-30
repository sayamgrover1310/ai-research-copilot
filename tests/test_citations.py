"""Unit tests for local-document citation records without Ollama."""

import unittest

from langchain_core.documents import Document

from src.rag.citations import build_citation_sources, format_citation


def make_document(
    *,
    source: str | None = "data/uploads/rag.pdf",
    source_filename: str | None = "rag.pdf",
    page_number: int | None = 3,
    chunk_index: int | None = 7,
    chunk_id: str | None = "chunk_7",
) -> Document:
    metadata = {
        key: value
        for key, value in {
            "source": source,
            "source_filename": source_filename,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
        }.items()
        if value is not None
    }
    return Document(page_content="Retrieved evidence.", metadata=metadata)


class CitationTests(unittest.TestCase):
    def test_builds_one_source_from_one_retrieved_document(self) -> None:
        sources = build_citation_sources([make_document()])

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].label, "[1]")
        self.assertEqual(sources[0].source, "data/uploads/rag.pdf")
        self.assertEqual(sources[0].source_filename, "rag.pdf")
        self.assertEqual(sources[0].page_number, 3)
        self.assertEqual(sources[0].chunk_indices, (7,))
        self.assertEqual(sources[0].chunk_ids, ("chunk_7",))
        self.assertEqual(format_citation(sources[0]), "[1] rag.pdf — Page 3")

    def test_keeps_multiple_files_or_pages_as_distinct_sources(self) -> None:
        sources = build_citation_sources(
            [
                make_document(page_number=3, chunk_id="chunk_7"),
                make_document(page_number=4, chunk_id="chunk_8"),
                make_document(
                    source="data/uploads/genai_notes.pdf",
                    source_filename="genai_notes.pdf",
                    page_number=7,
                    chunk_id="chunk_9",
                ),
            ]
        )

        self.assertEqual([source.label for source in sources], ["[1]", "[2]", "[3]"])
        self.assertEqual([source.page_number for source in sources], [3, 4, 7])
        self.assertEqual(sources[2].source_filename, "genai_notes.pdf")

    def test_deduplicates_chunks_from_the_same_source_and_page(self) -> None:
        sources = build_citation_sources(
            [
                make_document(chunk_index=7, chunk_id="chunk_7"),
                make_document(chunk_index=8, chunk_id="chunk_8"),
                make_document(chunk_index=7, chunk_id="chunk_7"),
            ]
        )

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].chunk_indices, (7, 8))
        self.assertEqual(sources[0].chunk_ids, ("chunk_7", "chunk_8"))

    def test_handles_missing_optional_metadata_without_crashing(self) -> None:
        filename_only = make_document(source=None, page_number=None, chunk_index=None, chunk_id=None)
        path_only = make_document(
            source="data/uploads/notes.txt",
            source_filename=None,
            page_number=None,
            chunk_index=None,
            chunk_id=None,
        )

        sources = build_citation_sources([filename_only, path_only])

        self.assertEqual(format_citation(sources[0]), "[1] rag.pdf")
        self.assertEqual(format_citation(sources[1]), "[2] notes.txt")
        self.assertEqual(sources[1].source_filename, None)
        self.assertEqual(sources[1].page_number, None)

    def test_numbering_is_deterministic_for_the_same_retrieval_order(self) -> None:
        documents = [
            make_document(source="data/uploads/b.pdf", source_filename="b.pdf", page_number=1),
            make_document(source="data/uploads/a.pdf", source_filename="a.pdf", page_number=2),
        ]

        self.assertEqual(build_citation_sources(documents), build_citation_sources(documents))
        self.assertEqual(
            [source.label for source in build_citation_sources(documents)],
            ["[1]", "[2]"],
        )


if __name__ == "__main__":
    unittest.main()
