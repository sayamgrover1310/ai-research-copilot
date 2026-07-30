"""Unit tests for LangGraph routing without Ollama, Chroma, or embeddings."""

import unittest

from langchain_core.documents import Document

from src.rag.basic_rag import RAGResult
from src.rag.citations import CitationSource
from src.web_research.basic_web_research import WebResearchResult
from src.web_research.deep_research import ResearchEvidence
from src.web_research.sources import WebSource
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


class FakeWebResearchFunction:
    def __init__(self) -> None:
        self.questions: list[str] = []
        self.source = WebSource(
            label="[W1]",
            title="RAG update",
            url="https://example.com/rag-update",
            content="A current RAG development.",
            rank=1,
        )

    def __call__(self, question: str, *args: object, **kwargs: object) -> WebResearchResult:
        self.questions.append(question)
        return WebResearchResult(
            answer="Current web evidence describes a RAG update.",
            web_sources=[self.source],
        )


class FakeResearchPlanner:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def __call__(self, question: str, *args: object, **kwargs: object) -> list[str]:
        self.questions.append(question)
        return ["RAG retrieval improvements", "RAG evaluation methods"]


class FailingResearchPlanner:
    def __call__(self, *args: object, **kwargs: object) -> list[str]:
        raise RuntimeError("planner unavailable")


class FakeResearchSearch:
    def __init__(self) -> None:
        self.plans: list[list[str]] = []
        self.source = WebSource(
            label="[W1]",
            title="RAG evidence",
            url="https://example.com/rag-evidence",
            content="Collected research evidence.",
            rank=1,
        )

    def __call__(self, research_queries: list[str]) -> tuple[list[ResearchEvidence], list[WebSource]]:
        self.plans.append(research_queries)
        return [ResearchEvidence(source=self.source, research_query=research_queries[0])], [self.source]


class FakeResearchSynthesis:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ResearchEvidence]]] = []

    def __call__(self, question: str, evidence: list[ResearchEvidence], **kwargs: object) -> str:
        self.calls.append((question, evidence))
        return "A synthesized Deep Research answer."


class DocumentWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vector_store = object()
        self.embedding_model = object()
        self.generation_model = FakeGenerationModel()
        self.rag_function = FakeRAGFunction()
        self.web_research_function = FakeWebResearchFunction()
        self.research_planner_function = FakeResearchPlanner()
        self.research_search_function = FakeResearchSearch()
        self.research_synthesis_function = FakeResearchSynthesis()

    def test_routes_explicit_document_question_to_document_rag(self) -> None:
        result = invoke_document_workflow(
            "According to my uploaded documents, what is RAG?",
            self.vector_store,
            self.embedding_model,
            generation_model=self.generation_model,
            rag_function=self.rag_function,
            web_research_function=self.web_research_function,
            research_planner_function=self.research_planner_function,
            research_search_function=self.research_search_function,
            research_synthesis_function=self.research_synthesis_function,
        )

        self.assertEqual(result["route"], "document_rag")
        self.assertEqual(result["answer"], "RAG uses retrieved knowledge.")
        self.assertEqual(result["source_documents"], [self.rag_function.document])
        self.assertEqual(result["sources"], [self.rag_function.source])
        self.assertEqual(result["web_sources"], [])
        self.assertEqual(len(self.rag_function.questions), 1)
        self.assertEqual(self.generation_model.prompts, [])

    def test_routes_general_question_to_general_node(self) -> None:
        result = invoke_document_workflow(
            "Write a short greeting.",
            self.vector_store,
            self.embedding_model,
            generation_model=self.generation_model,
            rag_function=self.rag_function,
            web_research_function=self.web_research_function,
            research_planner_function=self.research_planner_function,
            research_search_function=self.research_search_function,
            research_synthesis_function=self.research_synthesis_function,
        )

        self.assertEqual(result["route"], "general")
        self.assertEqual(result["answer"], "Hello! How can I help with your studies?")
        self.assertEqual(result["source_documents"], [])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["web_sources"], [])
        self.assertEqual(self.rag_function.questions, [])
        self.assertIn("Write a short greeting.", self.generation_model.prompts[0])

    def test_routes_current_question_to_web_research(self) -> None:
        result = invoke_document_workflow(
            "What are the latest developments in RAG?",
            self.vector_store,
            self.embedding_model,
            generation_model=self.generation_model,
            rag_function=self.rag_function,
            web_research_function=self.web_research_function,
            research_planner_function=self.research_planner_function,
            research_search_function=self.research_search_function,
            research_synthesis_function=self.research_synthesis_function,
        )

        self.assertEqual(result["route"], "web_research")
        self.assertEqual(result["answer"], "Current web evidence describes a RAG update.")
        self.assertEqual(result["source_documents"], [])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["web_sources"], [self.web_research_function.source])
        self.assertEqual(self.web_research_function.questions, ["What are the latest developments in RAG?"])

    def test_routes_research_request_through_planning_search_and_synthesis(self) -> None:
        question = "Research the major approaches, challenges, and recent developments in RAG."
        result = invoke_document_workflow(
            question,
            self.vector_store,
            self.embedding_model,
            generation_model=self.generation_model,
            rag_function=self.rag_function,
            web_research_function=self.web_research_function,
            research_planner_function=self.research_planner_function,
            research_search_function=self.research_search_function,
            research_synthesis_function=self.research_synthesis_function,
        )

        self.assertEqual(result["route"], "deep_research")
        self.assertEqual(result["research_queries"], ["RAG retrieval improvements", "RAG evaluation methods"])
        self.assertEqual(result["web_sources"], [self.research_search_function.source])
        self.assertEqual(result["web_evidence"][0].research_query, "RAG retrieval improvements")
        self.assertEqual(result["answer"], "A synthesized Deep Research answer.")
        self.assertEqual(self.research_planner_function.questions, [question])
        self.assertEqual(self.research_search_function.plans, [result["research_queries"]])
        self.assertEqual(len(self.research_synthesis_function.calls), 1)

    def test_ends_deep_research_when_planning_fails(self) -> None:
        result = invoke_document_workflow(
            "Research RAG challenges.",
            self.vector_store,
            self.embedding_model,
            generation_model=self.generation_model,
            rag_function=self.rag_function,
            web_research_function=self.web_research_function,
            research_planner_function=FailingResearchPlanner(),
            research_search_function=self.research_search_function,
            research_synthesis_function=self.research_synthesis_function,
        )

        self.assertEqual(result["route"], "deep_research")
        self.assertEqual(result["research_queries"], [])
        self.assertIn("planner unavailable", result["error"])
        self.assertEqual(self.research_search_function.plans, [])

    def test_router_is_deterministic(self) -> None:
        self.assertEqual(determine_route("Please summarize this PDF."), "document_rag")
        self.assertEqual(determine_route("What are the latest developments in RAG?"), "web_research")
        self.assertEqual(determine_route("Research the major approaches and challenges in RAG."), "deep_research")
        self.assertEqual(determine_route("Write a short greeting."), "general")

    def test_rejects_an_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            invoke_document_workflow(
                "   ",
                self.vector_store,
                self.embedding_model,
                generation_model=self.generation_model,
                rag_function=self.rag_function,
                web_research_function=self.web_research_function,
                research_planner_function=self.research_planner_function,
                research_search_function=self.research_search_function,
                research_synthesis_function=self.research_synthesis_function,
            )

    def test_general_prompt_is_small_and_contains_the_question(self) -> None:
        prompt = build_general_prompt("Write a short greeting.")
        self.assertIn("Write a short greeting.", prompt)
        self.assertIn("Reply briefly", prompt)


if __name__ == "__main__":
    unittest.main()
