"""Unit tests for Study Mode without Ollama, Chroma, or retrieval inference."""

import unittest

from langchain_core.documents import Document

from src.study.study_mode import StudyMode


def make_documents() -> list[Document]:
    return [
        Document(
            page_content="Retrieval Augmented Generation retrieves external knowledge before generating an answer.",
            metadata={
                "source": "data/uploads/rag.pdf",
                "source_filename": "rag.pdf",
                "page_number": 3,
                "chunk_index": 7,
                "chunk_id": "chunk_rag",
            },
        )
    ]


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeStructuredModel:
    def __init__(self, output: object) -> None:
        self.output = output
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return self.output


class FakeGenerationModel:
    def __init__(self, summary: str = "RAG retrieves external knowledge.") -> None:
        self.summary = summary
        self.prompts: list[str] = []
        self.structured_outputs: list[object] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.summary)

    def with_structured_output(self, _schema: object) -> FakeStructuredModel:
        return FakeStructuredModel(self.structured_outputs.pop(0))


class StudyModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[str] = []
        self.model = FakeGenerationModel()

        def fake_retriever(topic: str, *_args: object) -> list[Document]:
            self.calls.append(topic)
            return make_documents()

        self.study_mode = StudyMode(
            vector_store=object(),
            embedding_model=object(),
            generation_model=self.model,
            retriever=fake_retriever,
        )

    def test_summary_retrieves_evidence_and_preserves_sources(self) -> None:
        result = self.study_mode.summarize("Retrieval Augmented Generation")

        self.assertEqual(result.summary, "RAG retrieves external knowledge.")
        self.assertEqual(self.calls, ["Retrieval Augmented Generation"])
        self.assertEqual(result.source_documents, make_documents())
        self.assertEqual(result.sources[0].source_filename, "rag.pdf")
        self.assertIn(make_documents()[0].page_content, self.model.prompts[0])

    def test_generates_requested_valid_quiz(self) -> None:
        self.model.structured_outputs.append(
            {
                "questions": [
                    {
                        "question": "What does RAG retrieve?",
                        "options": ["External knowledge", "Only images", "Random text", "Nothing"],
                        "correct_answer": "External knowledge",
                        "explanation": "RAG retrieves external knowledge before generation.",
                    },
                    {
                        "question": "When does RAG retrieve knowledge?",
                        "options": ["Before generation", "After deployment", "Never", "Only at training"],
                        "correct_answer": "Before generation",
                        "explanation": "Retrieved knowledge is supplied before answering.",
                    },
                ]
            }
        )

        result = self.study_mode.generate_quiz("RAG", 2)

        self.assertEqual(len(result.questions), 2)
        self.assertTrue(all(len(question.options) == 4 for question in result.questions))
        self.assertTrue(all(question.correct_answer in question.options for question in result.questions))
        self.assertTrue(all(question.explanation for question in result.questions))
        self.assertEqual(result.sources[0].page_number, 3)

    def test_rejects_invalid_quiz_count_and_malformed_quiz(self) -> None:
        invalid_count = self.study_mode.generate_quiz("RAG", 11)
        self.model.structured_outputs.append(
            {
                "questions": [
                    {
                        "question": "Invalid question",
                        "options": ["A", "B", "C"],
                        "correct_answer": "A",
                        "explanation": "Too few options.",
                    }
                ]
            }
        )
        malformed = self.study_mode.generate_quiz("RAG", 1)

        self.assertIn("at most", invalid_count.error or "")
        self.assertEqual(invalid_count.questions, [])
        self.assertIn("four distinct", malformed.error or "")

    def test_generates_requested_valid_flashcards(self) -> None:
        self.model.structured_outputs.append(
            {
                "flashcards": [
                    {"front": "What is RAG?", "back": "A method that retrieves external knowledge before generation."},
                    {"front": "Why use RAG?", "back": "To ground answers in retrieved material."},
                ]
            }
        )

        result = self.study_mode.generate_flashcards("RAG", 2)

        self.assertEqual(len(result.flashcards), 2)
        self.assertTrue(all(card.front and card.back for card in result.flashcards))
        self.assertEqual(result.sources[0].chunk_ids, ("chunk_rag",))

    def test_rejects_invalid_flashcard_count(self) -> None:
        result = self.study_mode.generate_flashcards("RAG", 21)

        self.assertEqual(result.flashcards, [])
        self.assertIn("at most", result.error or "")

    def test_handles_empty_topic_no_evidence_and_generation_failure(self) -> None:
        empty_topic = self.study_mode.summarize("   ")
        failing_model = type(
            "FailingModel",
            (),
            {"invoke": lambda self, _prompt: (_ for _ in ()).throw(RuntimeError("Ollama unavailable"))},
        )()
        failing_summary = StudyMode(
            object(), object(), generation_model=failing_model, retriever=lambda *_args: make_documents()
        ).summarize("RAG")
        no_evidence = StudyMode(
            object(), object(), generation_model=self.model, retriever=lambda *_args: []
        ).generate_flashcards("RAG", 1)

        self.assertIn("topic must contain text", empty_topic.error or "")
        self.assertIn("Ollama unavailable", failing_summary.error or "")
        self.assertIn("could not find relevant material", no_evidence.error or "")


if __name__ == "__main__":
    unittest.main()
