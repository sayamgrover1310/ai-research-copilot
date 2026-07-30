"""Run the complete local Basic RAG flow with Ollama and Chroma."""

from langchain_core.documents import Document

from src.rag.basic_rag import answer_question, build_rag_prompt, get_generation_model
from src.rag.citations import format_citation
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.retriever import retrieve_documents
from src.retrieval.vector_store import get_vector_store, store_chunks


QUESTION = "What is Retrieval Augmented Generation?"


def make_demo_chunks() -> list[Document]:
    return [
        Document(
            page_content=(
                "Retrieval Augmented Generation retrieves external knowledge before "
                "generating an answer. This helps the answer use information from "
                "the provided documents."
            ),
            metadata={
                "source": "demo/rag.pdf",
                "source_filename": "rag.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Embeddings represent semantic meaning as numerical vectors.",
            metadata={
                "source": "demo/embeddings.pdf",
                "source_filename": "embeddings.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Photosynthesis converts light energy into chemical energy in plants.",
            metadata={
                "source": "demo/plants.pdf",
                "source_filename": "plants.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
        ),
    ]


def main() -> None:
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(
        embedding_model=embedding_model,
        workspace_id="basic_rag_demo",
    )
    store_chunks(make_demo_chunks(), vector_store)

    retrieved_documents = retrieve_documents(
        QUESTION,
        vector_store,
        embedding_model,
        top_k=2,
    )

    print(f"User question: {QUESTION}\n")
    print("Retrieved chunks:")
    for index, document in enumerate(retrieved_documents, start=1):
        print(f"{index}. {document.page_content}")

    print("\nPrompt sent to the generation model:\n")
    print(build_rag_prompt(QUESTION, retrieved_documents))

    result = answer_question(
        QUESTION,
        vector_store,
        embedding_model,
        generation_model=get_generation_model(),
        top_k=2,
    )
    if result.error:
        raise RuntimeError(result.error)

    print("\nGrounded answer:\n")
    print(result.answer)

    print("\nSources:\n")
    for source in result.sources:
        print(format_citation(source))


if __name__ == "__main__":
    main()
