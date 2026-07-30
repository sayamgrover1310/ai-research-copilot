"""Initial LangGraph routing between document RAG and a general response."""

from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph

from src.rag.basic_rag import RAGResult, answer_question, get_generation_model
from src.rag.citations import CitationSource
from src.web_research.basic_web_research import WebResearchResult, answer_web_question
from src.web_research.sources import WebSource


Route = Literal["document_rag", "web_research", "general"]
DOCUMENT_ROUTE_KEYWORDS = ("document", "uploaded", "upload", "pdf", "file", "my notes", "according to")
WEB_RESEARCH_ROUTE_KEYWORDS = ("latest", "current", "today", "news", "recent", "web", "online", "developments")


class WorkflowState(TypedDict, total=False):
    """The small amount of data shared by nodes in the initial workflow."""

    question: str
    route: Route
    answer: str
    source_documents: list[Document]
    sources: list[CitationSource]
    web_sources: list[WebSource]
    error: str | None


def determine_route(question: str) -> Route:
    """Route document and clearly current questions without an extra LLM call."""
    normalized_question = question.lower()
    if any(keyword in normalized_question for keyword in DOCUMENT_ROUTE_KEYWORDS):
        return "document_rag"
    if any(keyword in normalized_question for keyword in WEB_RESEARCH_ROUTE_KEYWORDS):
        return "web_research"
    return "general"


def create_document_workflow(
    vector_store: Chroma,
    embedding_model: Embeddings,
    generation_model: Any | None = None,
    rag_function: Callable[..., RAGResult] = answer_question,
    web_research_function: Callable[..., WebResearchResult] = answer_web_question,
) -> Any:
    """Compile a graph that routes to document RAG, web research, or general chat."""
    workflow = StateGraph(WorkflowState)

    def router_node(state: WorkflowState) -> WorkflowState:
        return {"route": determine_route(state["question"])}

    def document_rag_node(state: WorkflowState) -> WorkflowState:
        result = rag_function(
            state["question"],
            vector_store,
            embedding_model,
            generation_model=generation_model,
        )
        return {
            "answer": result.answer,
            "source_documents": result.source_documents,
            "sources": result.sources,
            "web_sources": [],
            "error": result.error,
        }

    def web_research_node(state: WorkflowState) -> WorkflowState:
        result = web_research_function(
            state["question"],
            generation_model=generation_model,
        )
        return {
            "answer": result.answer,
            "source_documents": [],
            "sources": [],
            "web_sources": result.web_sources,
            "error": result.error,
        }

    def general_node(state: WorkflowState) -> WorkflowState:
        model = generation_model or get_generation_model()
        prompt = build_general_prompt(state["question"])

        try:
            response = model.invoke(prompt)
            answer = _get_response_text(response)
        except Exception as error:
            return {
                "answer": "",
                "source_documents": [],
                "sources": [],
                "web_sources": [],
                "error": f"The local generation model could not produce a response: {error}",
            }

        return {
            "answer": answer,
            "source_documents": [],
            "sources": [],
            "web_sources": [],
            "error": None,
        }

    workflow.add_node("router", router_node)
    workflow.add_node("document_rag", document_rag_node)
    workflow.add_node("web_research", web_research_node)
    workflow.add_node("general", general_node)
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "document_rag": "document_rag",
            "web_research": "web_research",
            "general": "general",
        },
    )
    workflow.add_edge("document_rag", END)
    workflow.add_edge("web_research", END)
    workflow.add_edge("general", END)

    return workflow.compile()


def invoke_document_workflow(
    question: str,
    vector_store: Chroma,
    embedding_model: Embeddings,
    generation_model: Any | None = None,
    rag_function: Callable[..., RAGResult] = answer_question,
    web_research_function: Callable[..., WebResearchResult] = answer_web_question,
) -> WorkflowState:
    """Validate one question, compile the workflow, and return its final state."""
    if not question.strip():
        raise ValueError("A question must contain text before the workflow can run.")

    graph = create_document_workflow(
        vector_store,
        embedding_model,
        generation_model=generation_model,
        rag_function=rag_function,
        web_research_function=web_research_function,
    )
    return graph.invoke({"question": question})


def build_general_prompt(question: str) -> str:
    """Create the intentionally small prompt used by the general-response node."""
    return f"""You are a helpful research and study assistant.

Reply briefly and helpfully to the user's request.

User question:
{question}

Response:"""


def _get_response_text(response: Any) -> str:
    """Read plain text from a LangChain chat-model response."""
    content = getattr(response, "content", response)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The generation model returned an empty or non-text response.")
    return content.strip()
