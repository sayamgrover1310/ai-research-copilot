"""Verify a real local Document -> Chunk -> Ollama -> Chroma flow."""

from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import get_vector_store, store_chunks


def main() -> None:
    document = Document(
        page_content=(
            "RAG provides an LLM with external knowledge from documents. "
            "Vector databases store chunks so relevant evidence can be found later."
        ),
        metadata={
            "source": "data/uploads/rag.pdf",
            "source_filename": "rag.pdf",
            "file_type": "pdf",
            "page_number": 3,
        },
    )
    chunks = chunk_documents([document], chunk_size=80, chunk_overlap=20)

    embedding_model = get_embedding_model()
    vector_store = get_vector_store(
        embedding_model=embedding_model,
        workspace_id="integration_demo",
    )
    chunk_ids = store_chunks(chunks, vector_store)
    stored = vector_store.get(ids=chunk_ids, include=["embeddings", "documents", "metadatas"])

    print(f"Chunks stored: {len(chunk_ids)}")
    print(f"Embedding dimensions: {len(stored['embeddings'][0])}")
    print(f"First stored ID: {stored['ids'][0]}")
    print(f"First stored text: {stored['documents'][0]}")
    print(f"First stored metadata: {stored['metadatas'][0]}")


if __name__ == "__main__":
    main()
