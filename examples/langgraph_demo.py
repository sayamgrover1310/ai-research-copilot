"""Run the local LangGraph router with document RAG and general-response routes."""

from examples.basic_rag_demo import make_demo_chunks
from src.rag.citations import format_citation
from src.rag.basic_rag import get_generation_model
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import get_vector_store, store_chunks
from src.workflow.document_workflow import invoke_document_workflow


def print_result(question: str, result: dict) -> None:
    """Display the route, answer, and optional local-document sources."""
    print(f"Question: {question}")
    print(f"Selected route: {result['route']}")
    print(f"Answer: {result['answer']}")

    if result["sources"]:
        print("Sources:")
        for source in result["sources"]:
            print(format_citation(source))
    print()


def main() -> None:
    embedding_model = get_embedding_model()
    generation_model = get_generation_model()
    vector_store = get_vector_store(
        embedding_model=embedding_model,
        workspace_id="langgraph_demo",
    )
    store_chunks(make_demo_chunks(), vector_store)

    document_result = invoke_document_workflow(
        "According to my uploaded documents, what is RAG?",
        vector_store,
        embedding_model,
        generation_model=generation_model,
    )
    print_result("According to my uploaded documents, what is RAG?", document_result)

    general_result = invoke_document_workflow(
        "Write a short greeting.",
        vector_store,
        embedding_model,
        generation_model=generation_model,
    )
    print_result("Write a short greeting.", general_result)


if __name__ == "__main__":
    main()
