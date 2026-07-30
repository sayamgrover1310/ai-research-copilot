"""Unit tests for Basic RAG without a running Ollama server."""

import unittest

from langchain_core.documents import Document

from src.rag.basic_rag import (
    NO_CONTEXT_ANSWER,
    answer_question,
    build_context,
    build_rag_prompt,
)


def make_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "Retrieval Augmented Generation retrieves external knowledge before "
                "generating an answer."
            ),
            metadata={
                "source_filename": "rag.pdf",
                "page_number": 3,
                "chunk_index": 7,
                "chunk_id": "chunk_example",
            },
        )
    ]


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeGenerationModel:
    def __init__(self, response_text: str = "RAG uses retrieved external knowledge.") -> None:
        self.response_text = response_text
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.response_text)


class FailingGenerationModel:
    def invoke(self, prompt: str) -> FakeResponse:
        raise RuntimeError("Ollama is unavailable")


class FakeRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, object, object, int]] = []

    def __call__(
        self,
        question: str,
        vector_store: object,
        embedding_model: object,
        top_k: int,
    ) -> list[Document]:
        self.calls.append((question, vector_store, embedding_model, top_k))
        return self.documents


class BasicRAGTests(unittest.TestCase):
    def test_returns_answer_and_original_retrieved_documents(self) -> None:
        documents = make_documents()
        retriever = FakeRetriever(documents)
        model = FakeGenerationModel()

        result = answer_question(
            "What is Retrieval Augmented Generation?",
            vector_store=object(),
            embedding_model=object(),
            generation_model=model,
            top_k=2,
            retriever=retriever,
        )

        self.assertEqual(result.answer, "RAG uses retrieved external knowledge.")
        self.assertIs(result.source_documents[0], documents[0])
        self.assertEqual(result.sources[0].label, "[1]")
        self.assertEqual(result.sources[0].source_filename, "rag.pdf")
        self.assertIsNone(result.error)
        self.assertEqual(retriever.calls[0][3], 2)
        self.assertIn("What is Retrieval Augmented Generation?", model.prompts[0])
        self.assertIn(documents[0].page_content, model.prompts[0])

    def test_returns_a_safe_answer_without_calling_the_model_when_nothing_is_retrieved(self) -> None:
        model = FakeGenerationModel()

        result = answer_question(
            "What is RAG?",
            vector_store=object(),
            embedding_model=object(),
            generation_model=model,
            retriever=FakeRetriever([]),
        )

        self.assertEqual(result.answer, NO_CONTEXT_ANSWER)
        self.assertEqual(result.source_documents, [])
        self.assertEqual(result.sources, [])
        self.assertEqual(model.prompts, [])

    def test_reports_generation_failure_and_keeps_the_retrieved_evidence(self) -> None:
        documents = make_documents()

        result = answer_question(
            "What is RAG?",
            vector_store=object(),
            embedding_model=object(),
            generation_model=FailingGenerationModel(),
            retriever=FakeRetriever(documents),
        )

        self.assertEqual(result.answer, "")
        self.assertEqual(result.source_documents, documents)
        self.assertEqual(result.sources[0].chunk_ids, ("chunk_example",))
        self.assertIn("Ollama is unavailable", result.error or "")

    def test_rejects_an_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            answer_question(
                "   ",
                vector_store=object(),
                embedding_model=object(),
                generation_model=FakeGenerationModel(),
                retriever=FakeRetriever([]),
            )

    def test_build_context_and_prompt_use_chunk_text(self) -> None:
        documents = make_documents()
        context = build_context(documents)
        prompt = build_rag_prompt("What is RAG?", documents)

        self.assertEqual(context, f"[Retrieved chunk 1]\n{documents[0].page_content}")
        self.assertIn(context, prompt)
        self.assertIn("Do not invent facts", prompt)


if __name__ == "__main__":
    unittest.main()
