"""Unit tests for persistent Chroma storage without Ollama inference."""

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from src.retrieval.vector_store import create_chunk_id, get_vector_store, store_chunks


class FakeEmbeddingModel:
    """Predictable embedding model used only by vector-store unit tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


def make_chunk(text: str = "RAG uses external knowledge.") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "data/uploads/rag.pdf",
            "source_filename": "rag.pdf",
            "file_type": "pdf",
            "page_number": 3,
            "chunk_index": 7,
        },
    )


class VectorStoreTests(unittest.TestCase):
    def test_stores_chunk_text_metadata_and_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            vector_store = get_vector_store(
                embedding_model=FakeEmbeddingModel(),
                persist_directory=temporary_directory,
            )
            chunk = make_chunk()

            chunk_ids = store_chunks([chunk], vector_store)
            stored = vector_store.get(
                ids=chunk_ids,
                include=["documents", "embeddings", "metadatas"],
            )

        self.assertEqual(chunk_ids, [create_chunk_id(chunk)])
        self.assertEqual(stored["documents"], ["RAG uses external knowledge."])
        self.assertEqual(len(stored["embeddings"][0]), 3)
        self.assertEqual(stored["metadatas"][0]["source_filename"], "rag.pdf")
        self.assertEqual(stored["metadatas"][0]["page_number"], 3)
        self.assertEqual(stored["metadatas"][0]["chunk_index"], 7)
        self.assertEqual(stored["metadatas"][0]["chunk_id"], chunk_ids[0])
        self.assertEqual(chunk.metadata.get("chunk_id"), None)

    def test_duplicate_ingestion_does_not_create_a_second_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            vector_store = get_vector_store(
                embedding_model=FakeEmbeddingModel(),
                persist_directory=temporary_directory,
            )
            chunk = make_chunk()

            first_ids = store_chunks([chunk], vector_store)
            second_ids = store_chunks([chunk], vector_store)
            stored = vector_store.get()

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(stored["ids"], first_ids)

    def test_persists_data_across_new_vector_store_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first_store = get_vector_store(
                embedding_model=FakeEmbeddingModel(),
                persist_directory=Path(temporary_directory),
                workspace_id="research_a",
            )
            chunk_ids = store_chunks([make_chunk()], first_store)

            reopened_store = get_vector_store(
                embedding_model=FakeEmbeddingModel(),
                persist_directory=Path(temporary_directory),
                workspace_id="research_a",
            )
            stored = reopened_store.get(ids=chunk_ids)

        self.assertEqual(stored["ids"], chunk_ids)
        self.assertEqual(stored["documents"], ["RAG uses external knowledge."])

    def test_uses_separate_collections_for_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first_store = get_vector_store(
                embedding_model=FakeEmbeddingModel(),
                persist_directory=temporary_directory,
                workspace_id="workspace_one",
            )
            second_store = get_vector_store(
                embedding_model=FakeEmbeddingModel(),
                persist_directory=temporary_directory,
                workspace_id="workspace_two",
            )
            store_chunks([make_chunk()], first_store)

            second_store_records = second_store.get()

        self.assertEqual(second_store_records["ids"], [])

    def test_handles_empty_chunks_and_rejects_invalid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            vector_store = get_vector_store(
                embedding_model=FakeEmbeddingModel(),
                persist_directory=temporary_directory,
            )

            self.assertEqual(store_chunks([], vector_store), [])
            self.assertEqual(
                store_chunks([Document(page_content="  ", metadata={})], vector_store),
                [],
            )

            invalid_chunk = Document(
                page_content="Missing stable metadata.",
                metadata={"source_filename": "notes.txt"},
            )
            with self.assertRaises(ValueError):
                store_chunks([invalid_chunk], vector_store)


if __name__ == "__main__":
    unittest.main()
