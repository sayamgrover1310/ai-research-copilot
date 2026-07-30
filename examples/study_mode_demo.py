"""Run Study Mode using retrieved local RAG material and the local Ollama model."""

from langchain_core.documents import Document

from src.rag.basic_rag import get_generation_model
from src.rag.citations import format_citation
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import get_vector_store, store_chunks
from src.study.study_mode import StudyMode


TOPIC = "Retrieval Augmented Generation"


def make_demo_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "Retrieval Augmented Generation, or RAG, retrieves relevant external knowledge "
                "before an LLM generates an answer. This can ground answers in supplied documents "
                "and reduce unsupported responses when the retrieved evidence is relevant."
            ),
            metadata={
                "source": "demo/rag_study_notes.pdf",
                "source_filename": "rag_study_notes.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
        )
    ]


def main() -> None:
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(
        embedding_model=embedding_model,
        workspace_id="study_mode_demo",
    )
    store_chunks(make_demo_documents(), vector_store)
    study_mode = StudyMode(vector_store, embedding_model, get_generation_model())

    summary = study_mode.summarize(TOPIC)
    quiz = study_mode.generate_quiz(TOPIC, 3)
    flashcards = study_mode.generate_flashcards(TOPIC, 3)

    for result in (summary, quiz, flashcards):
        if result.error:
            raise RuntimeError(result.error)

    print("SUMMARY\n")
    print(summary.summary)
    print("\nSOURCES")
    for source in summary.sources:
        print(format_citation(source))

    print("\nQUIZ")
    for index, question in enumerate(quiz.questions, start=1):
        print(f"{index}. {question.question}")
        for option in question.options:
            print(f"   - {option}")
        print(f"   Correct answer: {question.correct_answer}")
        print(f"   Explanation: {question.explanation}")

    print("\nFLASHCARDS")
    for index, flashcard in enumerate(flashcards.flashcards, start=1):
        print(f"{index}. Front: {flashcard.front}")
        print(f"   Back: {flashcard.back}")


if __name__ == "__main__":
    main()
