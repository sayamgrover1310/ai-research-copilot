"""Tests for document chunking."""

import unittest

from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents


class ChunkDocumentsTests(unittest.TestCase):
    def test_splits_long_documents_and_preserves_metadata(self) -> None:
        document = Document(
            page_content="A" * 100 + "B" * 100,
            metadata={
                "source": "data/uploads/sample.pdf",
                "source_filename": "sample.pdf",
                "file_type": "pdf",
                "page_number": 1,
            },
        )

        chunks = chunk_documents([document], chunk_size=80, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertEqual([chunk.metadata["chunk_index"] for chunk in chunks], list(range(len(chunks))))
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "data/uploads/sample.pdf")
            self.assertEqual(chunk.metadata["source_filename"], "sample.pdf")
            self.assertEqual(chunk.metadata["file_type"], "pdf")
            self.assertEqual(chunk.metadata["page_number"], 1)

    def test_keeps_a_small_document_as_one_chunk(self) -> None:
        document = Document(
            page_content="Short study note.",
            metadata={"source_filename": "notes.txt", "file_type": "txt"},
        )

        chunks = chunk_documents([document])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "Short study note.")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[0].metadata["source_filename"], "notes.txt")

    def test_returns_an_empty_list_for_empty_input(self) -> None:
        self.assertEqual(chunk_documents([]), [])

    def test_ignores_documents_with_only_whitespace(self) -> None:
        document = Document(page_content="  \n", metadata={"source_filename": "empty.txt"})

        self.assertEqual(chunk_documents([document]), [])

    def test_rejects_invalid_chunk_settings(self) -> None:
        document = Document(page_content="Study note", metadata={})

        with self.assertRaises(ValueError):
            chunk_documents([document], chunk_size=0)

        with self.assertRaises(ValueError):
            chunk_documents([document], chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()
