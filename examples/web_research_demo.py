"""Run one real Tavily search through the LangGraph web-research route."""

from src.rag.basic_rag import get_generation_model
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import get_vector_store
from src.web_research.sources import format_web_source
from src.workflow.document_workflow import invoke_document_workflow


QUESTION = "What are the latest developments in RAG?"


def main() -> None:
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(
        embedding_model=embedding_model,
        workspace_id="web_research_demo",
    )
    result = invoke_document_workflow(
        QUESTION,
        vector_store,
        embedding_model,
        generation_model=get_generation_model(),
    )

    if result.get("error"):
        raise RuntimeError(result["error"])

    print(f"Question: {QUESTION}")
    print(f"Selected route: {result['route']}")
    print("\nWeb sources used as evidence:")
    for source in result["web_sources"]:
        print(format_web_source(source))
    print("\nGrounded answer:")
    print(result["answer"])


if __name__ == "__main__":
    main()
