"""Generate grounded answers from documents returned by semantic retrieval."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama

from src.config import get_generation_model_name, get_ollama_base_url
from src.rag.citations import CitationSource, build_citation_sources
from src.retrieval.retriever import DEFAULT_TOP_K, retrieve_documents


NO_CONTEXT_ANSWER = (
    "I could not find relevant information in the uploaded documents to answer that question."
)


@dataclass
class RAGResult:
    """The generated answer, retrieved evidence, and application-built sources."""

    answer: str
    source_documents: list[Document]
    sources: list[CitationSource]
    error: str | None = None


def get_generation_model() -> ChatOllama:
    """Create the configured local Ollama chat model for answer generation."""
    return ChatOllama(
        model=get_generation_model_name(),
        base_url=get_ollama_base_url(),
        temperature=0,
    )


def answer_question(
    question: str,
    vector_store: Chroma,
    embedding_model: Embeddings,
    generation_model: Any | None = None,
    top_k: int = DEFAULT_TOP_K,
    retriever: Callable[[str, Chroma, Embeddings, int], list[Document]] = retrieve_documents,
) -> RAGResult:
    """Retrieve evidence for a question and generate an answer grounded in that evidence."""
    if not question.strip():
        raise ValueError("A question must contain text before it can be answered.")

    source_documents = retriever(question, vector_store, embedding_model, top_k)
    if not source_documents:
        return RAGResult(answer=NO_CONTEXT_ANSWER, source_documents=[], sources=[])

    sources = build_citation_sources(source_documents)

    prompt = build_rag_prompt(question, source_documents)
    model = generation_model or get_generation_model()

    try:
        response = model.invoke(prompt)
        answer = _get_response_text(response)
    except Exception as error:
        return RAGResult(
            answer="",
            source_documents=source_documents,
            sources=sources,
            error=f"The local generation model could not produce an answer: {error}",
        )

    return RAGResult(answer=answer, source_documents=source_documents, sources=sources)


def build_context(documents: list[Document]) -> str:
    """Combine retrieved chunk text into the context supplied to the generation model."""
    return "\n\n".join(
        f"[Retrieved chunk {index}]\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )


def build_rag_prompt(question: str, documents: list[Document]) -> str:
    """Build a prompt that asks the model to use retrieved evidence only."""
    context = build_context(documents)
    return f"""You are a helpful research and study assistant.

Answer the user's question using only the retrieved context below.
If the context does not contain enough information, say that you do not have enough information in the uploaded documents.
Do not invent facts or use outside knowledge.

User question:
{question}

Retrieved context:
{context}

Answer:"""


def _get_response_text(response: Any) -> str:
    """Read text from LangChain's chat-model response and reject empty output."""
    content = getattr(response, "content", response)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The generation model returned an empty or non-text response.")
    return content.strip()
