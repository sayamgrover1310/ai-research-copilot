"""Run a real local Ollama semantic-similarity demonstration."""

from langchain_core.documents import Document

from src.retrieval.embeddings import (
    cosine_similarity,
    embed_chunks,
    get_embedding_model,
)


def main() -> None:
    sentences = [
        "The student is learning artificial intelligence.",
        "The learner is studying AI and machine learning.",
        "The weather is rainy today.",
    ]
    documents = [Document(page_content=sentence, metadata={}) for sentence in sentences]

    embedding_model = get_embedding_model()
    vectors = embed_chunks(documents, embedding_model)

    related_score = cosine_similarity(vectors[0], vectors[1])
    unrelated_score = cosine_similarity(vectors[0], vectors[2])

    print(f"Embedding dimensions: {len(vectors[0])}")
    print(f"Sentence 1 vs sentence 2: {related_score:.4f}")
    print(f"Sentence 1 vs sentence 3: {unrelated_score:.4f}")

    if related_score <= unrelated_score:
        raise RuntimeError("Expected the related sentences to have the higher similarity score.")

    print("Verification passed: the related sentences have the higher similarity score.")


if __name__ == "__main__":
    main()
