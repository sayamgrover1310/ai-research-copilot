"""Unit tests for one-step web research without Ollama or internet access."""

import unittest

from src.web_research.basic_web_research import (
    NO_WEB_EVIDENCE_ANSWER,
    answer_web_question,
    build_web_context,
    build_web_research_prompt,
)
from src.web_research.sources import WebSource
from src.web_research.tavily_search import WebSearchError


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeGenerationModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse("The retrieved web evidence describes a recent RAG development.")


def make_sources() -> list[WebSource]:
    return [
        WebSource(
            label="[W1]",
            title="RAG update",
            url="https://example.com/rag",
            content="A recent RAG development was published.",
            rank=1,
        )
    ]


class BasicWebResearchTests(unittest.TestCase):
    def test_returns_answer_and_provider_sources(self) -> None:
        model = FakeGenerationModel()
        calls: list[tuple[str, int | None]] = []

        def fake_search(question: str, max_results: int | None) -> list[WebSource]:
            calls.append((question, max_results))
            return make_sources()

        result = answer_web_question(
            "What are the latest developments in RAG?",
            generation_model=model,
            max_results=2,
            search_function=fake_search,
        )

        self.assertEqual(result.answer, "The retrieved web evidence describes a recent RAG development.")
        self.assertEqual(result.web_sources, make_sources())
        self.assertIsNone(result.error)
        self.assertEqual(calls, [("What are the latest developments in RAG?", 2)])
        self.assertIn("https://example.com/rag", model.prompts[0])

    def test_returns_safe_answer_without_calling_model_when_search_is_empty(self) -> None:
        model = FakeGenerationModel()
        result = answer_web_question(
            "What is current?",
            generation_model=model,
            search_function=lambda _question, _limit: [],
        )

        self.assertEqual(result.answer, NO_WEB_EVIDENCE_ANSWER)
        self.assertEqual(result.web_sources, [])
        self.assertEqual(model.prompts, [])

    def test_reports_search_and_generation_failures(self) -> None:
        search_failure = answer_web_question(
            "Latest RAG news",
            search_function=lambda _question, _limit: (_ for _ in ()).throw(WebSearchError("key missing")),
        )
        generation_failure = answer_web_question(
            "Latest RAG news",
            generation_model=type("FailingModel", (), {"invoke": lambda self, prompt: (_ for _ in ()).throw(RuntimeError("Ollama unavailable"))})(),
            search_function=lambda _question, _limit: make_sources(),
        )

        self.assertIn("key missing", search_failure.error or "")
        self.assertEqual(search_failure.web_sources, [])
        self.assertIn("Ollama unavailable", generation_failure.error or "")
        self.assertEqual(generation_failure.web_sources, make_sources())

    def test_rejects_empty_question_and_builds_grounded_context(self) -> None:
        with self.assertRaises(ValueError):
            answer_web_question("   ", search_function=lambda _question, _limit: [])

        context = build_web_context(make_sources())
        prompt = build_web_research_prompt("Latest RAG news", make_sources())
        self.assertIn("[W1] RAG update", context)
        self.assertIn(context, prompt)
        self.assertIn("Do not invent facts", prompt)


if __name__ == "__main__":
    unittest.main()
