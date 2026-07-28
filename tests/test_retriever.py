"""Unit tests for semantic retrieval without Ollama inference."""

import tempfile
import unittest

from langchain_core.documents import Document

from src.retrieval.retriever import (
    retrieve_documents,
    retrieve_documents_with_scores,
)
from src.retrieval.vector_store import get_vector_store, store_chunks


class FakeSemanticEmbeddingModel:
    """Returns simple fixed vectors so retrieval ranking is predictable in tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)

    def _vector_for(self, text: str) -> list[float]:
        normalized_text = text.lower()
        if "retrieval augmented generation" in normalized_text or "external information" in normalized_text:
            return [1.0, 0.0]
        if "embeddings" in normalized_text:
            return [0.5, 0.5]
        return [0.0, 1.0]


def make_chunks() -> list[Document]:
    texts = [
        "Retrieval Augmented Generation retrieves external knowledge before generating an answer.",
        "Embeddings represent semantic meaning as numerical vectors.",
        "Photosynthesis allows plants to convert light energy into chemical energy.",
    ]
    return [
        Document(
            page_content=text,
            metadata={
                "source": f"data/uploads/topic_{index}.pdf",
                "source_filename": f"topic_{index}.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": index,
            },
        )
        for index, text in enumerate(texts)
    ]


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.embedding_model = FakeSemanticEmbeddingModel()
        self.vector_store = get_vector_store(
            embedding_model=self.embedding_model,
            persist_directory=self.temporary_directory.name,
            workspace_id="retrieval_test",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_returns_rag_chunk_first_and_preserves_metadata(self) -> None:
        store_chunks(make_chunks(), self.vector_store)

        results = retrieve_documents(
            "How does RAG use external information?",
            self.vector_store,
            self.embedding_model,
            top_k=2,
        )

        self.assertEqual(len(results), 2)
        self.assertIn("Retrieval Augmented Generation", results[0].page_content)
        self.assertEqual(results[0].metadata["source_filename"], "topic_0.pdf")
        self.assertEqual(results[0].metadata["page_number"], 1)
        self.assertEqual(results[0].metadata["chunk_index"], 0)
        self.assertIn("chunk_id", results[0].metadata)

    def test_returns_relevance_scores_in_descending_order(self) -> None:
        store_chunks(make_chunks(), self.vector_store)

        results = retrieve_documents_with_scores(
            "How does RAG use external information?",
            self.vector_store,
            self.embedding_model,
            top_k=3,
        )

        self.assertEqual(len(results), 3)
        self.assertGreaterEqual(results[0][1], results[1][1])
        self.assertGreaterEqual(results[1][1], results[2][1])

    def test_returns_empty_list_for_an_empty_vector_store(self) -> None:
        results = retrieve_documents(
            "How does RAG use external information?",
            self.vector_store,
            self.embedding_model,
        )

        self.assertEqual(results, [])

    def test_returns_fewer_results_when_fewer_chunks_are_stored(self) -> None:
        store_chunks([make_chunks()[0]], self.vector_store)

        results = retrieve_documents(
            "How does RAG use external information?",
            self.vector_store,
            self.embedding_model,
            top_k=4,
        )

        self.assertEqual(len(results), 1)

    def test_rejects_empty_query_and_invalid_top_k(self) -> None:
        with self.assertRaises(ValueError):
            retrieve_documents("  ", self.vector_store, self.embedding_model)

        with self.assertRaises(ValueError):
            retrieve_documents("RAG", self.vector_store, self.embedding_model, top_k=0)


if __name__ == "__main__":
    unittest.main()
