"""Unit tests for LangGraph routing without Ollama, Chroma, or embeddings."""

import unittest

from langchain_core.documents import Document

from src.rag.basic_rag import RAGResult
from src.rag.citations import CitationSource
from src.workflow.document_workflow import (
    build_general_prompt,
    determine_route,
    invoke_document_workflow,
)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeGenerationModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse("Hello! How can I help with your studies?")


class FakeRAGFunction:
    def __init__(self) -> None:
        self.questions: list[str] = []
        self.document = Document(
            page_content="RAG retrieves external knowledge before generating an answer.",
            metadata={"source_filename": "rag.pdf", "page_number": 3, "chunk_id": "chunk_1"},
        )
        self.source = CitationSource(
            label="[1]",
            source="data/uploads/rag.pdf",
            source_filename="rag.pdf",
            page_number=3,
            chunk_indices=(7,),
            chunk_ids=("chunk_1",),
        )

    def __call__(self, question: str, *args: object, **kwargs: object) -> RAGResult:
        self.questions.append(question)
        return RAGResult(
            answer="RAG uses retrieved knowledge.",
            source_documents=[self.document],
            sources=[self.source],
        )


class DocumentWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vector_store = object()
        self.embedding_model = object()
        self.generation_model = FakeGenerationModel()
        self.rag_function = FakeRAGFunction()

    def test_routes_explicit_document_question_to_document_rag(self) -> None:
        result = invoke_document_workflow(
            "According to my uploaded documents, what is RAG?",
            self.vector_store,
            self.embedding_model,
            generation_model=self.generation_model,
            rag_function=self.rag_function,
        )

        self.assertEqual(result["route"], "document_rag")
        self.assertEqual(result["answer"], "RAG uses retrieved knowledge.")
        self.assertEqual(result["source_documents"], [self.rag_function.document])
        self.assertEqual(result["sources"], [self.rag_function.source])
        self.assertEqual(len(self.rag_function.questions), 1)
        self.assertEqual(self.generation_model.prompts, [])

    def test_routes_general_question_to_general_node(self) -> None:
        result = invoke_document_workflow(
            "Write a short greeting.",
            self.vector_store,
            self.embedding_model,
            generation_model=self.generation_model,
            rag_function=self.rag_function,
        )

        self.assertEqual(result["route"], "general")
        self.assertEqual(result["answer"], "Hello! How can I help with your studies?")
        self.assertEqual(result["source_documents"], [])
        self.assertEqual(result["sources"], [])
        self.assertEqual(self.rag_function.questions, [])
        self.assertIn("Write a short greeting.", self.generation_model.prompts[0])

    def test_router_is_deterministic(self) -> None:
        self.assertEqual(determine_route("Please summarize this PDF."), "document_rag")
        self.assertEqual(determine_route("Write a short greeting."), "general")

    def test_rejects_an_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            invoke_document_workflow(
                "   ",
                self.vector_store,
                self.embedding_model,
                generation_model=self.generation_model,
                rag_function=self.rag_function,
            )

    def test_general_prompt_is_small_and_contains_the_question(self) -> None:
        prompt = build_general_prompt("Write a short greeting.")
        self.assertIn("Write a short greeting.", prompt)
        self.assertIn("Reply briefly", prompt)


if __name__ == "__main__":
    unittest.main()
