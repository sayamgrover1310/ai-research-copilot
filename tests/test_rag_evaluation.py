"""Unit tests for deterministic document RAG evaluation logic."""

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from src.evaluation.rag_evaluation import (
    EvaluationExample,
    calculate_human_rating_averages,
    evaluate_retrieval,
    load_evaluation_dataset,
    render_retrieval_report,
)


def fake_retriever(_query, _store, _embedding_model, _top_k):
    results = [
        (Document(page_content="Unrelated", metadata={"source_filename": "other.pdf", "page_number": 1}), 0.9),
        (Document(page_content="Relevant later", metadata={"source_filename": "rag.pdf", "page_number": 3}), 0.8),
    ]
    return results[:_top_k]


class RAGEvaluationTests(unittest.TestCase):
    def test_calculates_hit_rate_and_mrr_for_rank_one_rank_later_and_absent(self) -> None:
        examples = [
            EvaluationExample("rank one", "rag.pdf", 3),
            EvaluationExample("rank later", "rag.pdf", 3),
            EvaluationExample("absent", "missing.pdf", 1),
        ]

        def retriever(query, store, embedding_model, top_k):
            if query == "rank one":
                return [(
                    Document(page_content="Relevant", metadata={"source_filename": "rag.pdf", "page_number": 3}),
                    0.9,
                )]
            if query == "rank later":
                return fake_retriever(query, store, embedding_model, top_k)
            return [(
                Document(page_content="Wrong", metadata={"source_filename": "other.pdf", "page_number": 1}),
                0.9,
            )]

        report = evaluate_retrieval(examples, object(), object(), (1, 3, 5), retriever)

        self.assertEqual(report.hit_rates, {1: 1 / 3, 3: 2 / 3, 5: 2 / 3})
        self.assertAlmostEqual(report.mrr or 0, (1 + 1 / 2) / 3)
        self.assertEqual(report.question_results[0].first_relevant_rank, 1)
        self.assertEqual(report.question_results[1].first_relevant_rank, 2)
        self.assertIsNone(report.question_results[2].first_relevant_rank)

    def test_handles_an_empty_dataset_and_rejects_invalid_k_values(self) -> None:
        report = evaluate_retrieval([], object(), object(), (1,), fake_retriever)
        self.assertEqual(report.question_count, 0)
        self.assertEqual(report.retrieval_question_count, 0)
        self.assertEqual(report.hit_rates, {1: 0.0})
        self.assertIsNone(report.mrr)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            evaluate_retrieval([], object(), object(), (0,), fake_retriever)
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_retrieval([], object(), object(), (), fake_retriever)

    def test_ignores_no_answer_examples_for_retrieval_metrics_and_renders_failures(self) -> None:
        examples = [
            EvaluationExample("known", "rag.pdf", 3),
            EvaluationExample("unknown", None, expects_no_answer=True),
        ]
        report = evaluate_retrieval(examples, object(), object(), (1,), fake_retriever)

        self.assertEqual(report.retrieval_question_count, 1)
        self.assertEqual(report.hit_rates, {1: 0.0})
        output = render_retrieval_report(report)
        self.assertIn("Failed retrieval cases: 1", output)
        self.assertIn("'known'", output)

    def test_loads_dataset_and_calculates_optional_human_rating_averages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "examples.json"
            path.write_text(json.dumps([
                {"question": "Question", "expected_source": "notes.pdf", "groundedness_rating": 2, "answer_relevance_rating": 1},
                {"question": "No answer", "expects_no_answer": True, "groundedness_rating": 0, "answer_relevance_rating": 2},
            ]), encoding="utf-8")
            examples = load_evaluation_dataset(path)

        self.assertEqual(calculate_human_rating_averages(examples), {"groundedness": 1.0, "answer_relevance": 1.5})


if __name__ == "__main__":
    unittest.main()
