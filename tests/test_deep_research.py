"""Unit tests for bounded Deep Research without Ollama or web access."""

import unittest

from src.web_research.deep_research import (
    ResearchPlanningError,
    ResearchSearchError,
    ResearchSynthesisError,
    build_research_context,
    build_research_synthesis_prompt,
    collect_research_evidence,
    normalize_research_queries,
    plan_research,
    synthesize_research,
)
from src.web_research.sources import WebSource


def make_source(url: str, title: str = "Source", content: str = "Evidence.") -> WebSource:
    return WebSource(label="[W1]", title=title, url=url, content=content, rank=1)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeGenerationModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse("A synthesis based on the collected evidence.")


class FakeStructuredPlanner:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> dict[str, list[str]]:
        self.prompts.append(prompt)
        return {"research_queries": ["RAG retrieval", "RAG evaluation", "RAG challenges"]}


class FakePlannerModel:
    def __init__(self) -> None:
        self.schema: object | None = None
        self.planner = FakeStructuredPlanner()

    def with_structured_output(self, schema: object) -> FakeStructuredPlanner:
        self.schema = schema
        return self.planner


class DeepResearchTests(unittest.TestCase):
    def test_planner_uses_structured_output_and_bounds_queries(self) -> None:
        model = FakePlannerModel()

        queries = plan_research("Research RAG", generation_model=model, max_queries=2)

        self.assertEqual(queries, ["RAG retrieval", "RAG evaluation"])
        self.assertIsNotNone(model.schema)
        self.assertIn("Return between 2 and 2", model.planner.prompts[0])

    def test_reports_structured_planner_failure(self) -> None:
        failing_model = type(
            "FailingPlannerModel",
            (),
            {"with_structured_output": lambda self, _schema: (_ for _ in ()).throw(RuntimeError("bad JSON"))},
        )()

        with self.assertRaises(ResearchPlanningError):
            plan_research("Research RAG", generation_model=failing_model)

    def test_normalizes_a_bounded_unique_research_plan(self) -> None:
        queries = normalize_research_queries(
            ["RAG retrieval improvements", "RAG challenges", "RAG challenges", "RAG evaluation"],
            max_queries=3,
        )

        self.assertEqual(queries, ["RAG retrieval improvements", "RAG challenges", "RAG evaluation"])

    def test_rejects_a_plan_with_fewer_than_two_queries(self) -> None:
        with self.assertRaises(ResearchPlanningError):
            normalize_research_queries(["Only one query"], max_queries=4)

    def test_searches_each_query_and_deduplicates_urls(self) -> None:
        calls: list[tuple[str, int | None]] = []

        def fake_search(query: str, max_results: int | None) -> list[WebSource]:
            calls.append((query, max_results))
            if query == "retrieval":
                return [make_source("https://example.com/shared"), make_source("https://example.com/a")]
            return [make_source("https://example.com/shared"), make_source("https://example.com/b")]

        evidence, sources = collect_research_evidence(
            ["retrieval", "evaluation"],
            search_function=fake_search,
            results_per_query=2,
            max_evidence=3,
            max_content_chars=20,
        )

        self.assertEqual(calls, [("retrieval", 2), ("evaluation", 2)])
        self.assertEqual([source.url for source in sources], [
            "https://example.com/shared",
            "https://example.com/a",
            "https://example.com/b",
        ])
        self.assertEqual([item.research_query for item in evidence], ["retrieval", "retrieval", "evaluation"])
        self.assertEqual([source.label for source in sources], ["[W1]", "[W2]", "[W3]"])

    def test_reports_search_failure_when_every_search_fails(self) -> None:
        with self.assertRaises(ResearchSearchError):
            collect_research_evidence(
                ["one", "two"],
                search_function=lambda _query, _limit: (_ for _ in ()).throw(RuntimeError("provider down")),
            )

    def test_synthesis_receives_collected_evidence(self) -> None:
        evidence, _sources = collect_research_evidence(
            ["retrieval"],
            search_function=lambda _query, _limit: [make_source("https://example.com/rag", content="RAG evidence")],
        )
        model = FakeGenerationModel()

        answer = synthesize_research("What is RAG?", evidence, generation_model=model)

        self.assertEqual(answer, "A synthesis based on the collected evidence.")
        self.assertIn("https://example.com/rag", model.prompts[0])
        self.assertIn("Found for research query: retrieval", build_research_context(evidence))
        self.assertIn("Do not invent facts", build_research_synthesis_prompt("What is RAG?", evidence))

    def test_reports_synthesis_failure(self) -> None:
        evidence, _sources = collect_research_evidence(
            ["retrieval"],
            search_function=lambda _query, _limit: [make_source("https://example.com/rag")],
        )
        failing_model = type(
            "FailingModel",
            (),
            {"invoke": lambda self, _prompt: (_ for _ in ()).throw(RuntimeError("Ollama unavailable"))},
        )()

        with self.assertRaises(ResearchSynthesisError):
            synthesize_research("What is RAG?", evidence, generation_model=failing_model)


if __name__ == "__main__":
    unittest.main()
