"""Run the bounded Deep Research route with local Ollama and Tavily."""

from src.rag.basic_rag import get_generation_model
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import get_vector_store
from src.web_research.sources import format_web_source
from src.workflow.document_workflow import invoke_document_workflow


QUESTION = "Research the major challenges and current approaches in Retrieval Augmented Generation."


def main() -> None:
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(
        embedding_model=embedding_model,
        workspace_id="deep_research_demo",
    )
    result = invoke_document_workflow(
        QUESTION,
        vector_store,
        embedding_model,
        generation_model=get_generation_model(),
    )

    if result.get("error"):
        raise RuntimeError(result["error"])

    print("ORIGINAL QUESTION\n")
    print(QUESTION)
    print("\nRESEARCH PLAN")
    for index, query in enumerate(result["research_queries"], start=1):
        print(f"{index}. {query}")
    print("\nEVIDENCE/SOURCES")
    for evidence in result["web_evidence"]:
        print(f"{format_web_source(evidence.source)}")
        print(f"  Found for: {evidence.research_query}")
    print("\nFINAL SYNTHESIS\n")
    print(result["answer"])


if __name__ == "__main__":
    main()
