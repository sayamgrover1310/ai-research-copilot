"""Run real local Ollama and Chroma semantic retrieval without generating an answer."""

from langchain_core.documents import Document

from src.retrieval.embeddings import get_embedding_model
from src.retrieval.retriever import retrieve_documents_with_scores
from src.retrieval.vector_store import get_vector_store, store_chunks


def main() -> None:
    chunks = [
        Document(
            page_content=(
                "Retrieval Augmented Generation retrieves external knowledge before "
                "generating an answer."
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
                "chunk_index": 1,
            },
        ),
        Document(
            page_content=(
                "Photosynthesis allows plants to convert light energy into chemical energy."
            ),
            metadata={
                "source": "demo/photosynthesis.pdf",
                "source_filename": "photosynthesis.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 2,
            },
        ),
    ]
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(
        embedding_model=embedding_model,
        workspace_id="semantic_retrieval_demo",
    )
    store_chunks(chunks, vector_store)

    results = retrieve_documents_with_scores(
        "How does RAG use external information?",
        vector_store,
        embedding_model,
        top_k=3,
    )

    for rank, (document, score) in enumerate(results, start=1):
        print(f"{rank}. score={score:.4f} | {document.page_content}")

    if "Retrieval Augmented Generation" not in results[0][0].page_content:
        raise RuntimeError("Expected the RAG chunk to rank first.")

    print("Verification passed: the RAG chunk ranked first.")


if __name__ == "__main__":
    main()
