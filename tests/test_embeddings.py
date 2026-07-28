"""Unit tests for the Ollama embedding layer without requiring Ollama."""

import unittest

from langchain_core.documents import Document

from src.retrieval.embeddings import cosine_similarity, embed_chunks, embed_query


class FakeEmbeddingModel:
    """Predictable stand-in for an Ollama embedding client during unit tests."""

    def __init__(self) -> None:
        self.document_texts: list[str] = []
        self.query_text: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_texts = texts
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        self.query_text = text
        return [float(len(text)), 1.0]


class EmbeddingTests(unittest.TestCase):
    def test_embeds_non_empty_chunk_text_in_order(self) -> None:
        chunks = [
            Document(page_content="First chunk", metadata={"chunk_index": 0}),
            Document(page_content="Second chunk", metadata={"chunk_index": 1}),
        ]
        embedding_model = FakeEmbeddingModel()

        vectors = embed_chunks(chunks, embedding_model)

        self.assertEqual(embedding_model.document_texts, ["First chunk", "Second chunk"])
        self.assertEqual(vectors, [[0.0, 11.0], [1.0, 12.0]])
        self.assertEqual(chunks[0].metadata, {"chunk_index": 0})

    def test_returns_an_empty_list_when_no_chunks_contain_text(self) -> None:
        embedding_model = FakeEmbeddingModel()
        chunks = [Document(page_content="  \n", metadata={})]

        self.assertEqual(embed_chunks(chunks, embedding_model), [])
        self.assertEqual(embedding_model.document_texts, [])

    def test_embeds_a_query(self) -> None:
        embedding_model = FakeEmbeddingModel()

        vector = embed_query("What is machine learning?", embedding_model)

        self.assertEqual(embedding_model.query_text, "What is machine learning?")
        self.assertEqual(vector, [25.0, 1.0])

    def test_rejects_an_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            embed_query("   ", FakeEmbeddingModel())

    def test_cosine_similarity_ranks_related_vectors_higher(self) -> None:
        reference = [1.0, 0.0]
        related = [0.9, 0.1]
        unrelated = [0.0, 1.0]

        self.assertGreater(
            cosine_similarity(reference, related),
            cosine_similarity(reference, unrelated),
        )


if __name__ == "__main__":
    unittest.main()
