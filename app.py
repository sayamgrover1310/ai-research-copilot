"""Streamlit interface for the existing AI Research & Study Copilot backend."""

from typing import Any

import streamlit as st

from src.config import get_study_flashcard_max_count, get_study_quiz_max_questions
from src.rag.basic_rag import answer_question, get_generation_model
from src.rag.citations import CitationSource, format_citation
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import get_vector_store
from src.services.workspace_service import (
    UploadedDocumentData,
    get_workspace_document_count,
    process_uploaded_documents,
)
from src.study.study_mode import FlashcardResult, QuizResult, StudyMode, TopicSummaryResult
from src.web_research.basic_web_research import WebResearchResult, answer_web_question
from src.web_research.sources import WebSource, format_web_source
from src.workflow.document_workflow import invoke_document_workflow


st.set_page_config(page_title="AI Research & Study Copilot", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_workspace_resources(workspace_id: str) -> tuple[Any, Any]:
    """Cache reusable local clients per workspace, not generated answers or user input."""
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(embedding_model=embedding_model, workspace_id=workspace_id)
    return embedding_model, vector_store


@st.cache_resource(show_spinner=False)
def get_cached_generation_model() -> Any:
    """Cache the local Ollama chat client shared by UI actions."""
    return get_generation_model()


def initialize_session_state(workspace_id: str) -> None:
    """Clear mode results when the selected workspace changes."""
    if st.session_state.get("active_workspace") != workspace_id:
        st.session_state.active_workspace = workspace_id
        st.session_state.ask_result = None
        st.session_state.web_result = None
        st.session_state.deep_result = None
        st.session_state.study_summary_result = None
        st.session_state.study_quiz_result = None
        st.session_state.study_flashcard_result = None
        st.session_state.last_processing = None


def show_document_sources(sources: list[CitationSource], documents: list[Any]) -> None:
    """Render application-built local citations and optional retrieved evidence."""
    if not sources:
        return

    st.subheader("Sources")
    for source in sources:
        st.markdown(format_citation(source))
        matching_documents = [
            document
            for document in documents
            if document.metadata.get("chunk_id") in source.chunk_ids
        ]
        if matching_documents:
            with st.expander(f"View retrieved evidence for {source.label}"):
                for document in matching_documents:
                    st.write(document.page_content)


def show_web_sources(sources: list[WebSource]) -> None:
    """Render provider-derived web titles and URLs without generating metadata in the UI."""
    if not sources:
        return

    st.subheader("Sources")
    for source in sources:
        st.markdown(f"{source.label} [{source.title}]({source.url})")
        with st.expander("View retrieved web evidence"):
            st.write(source.content)
            st.caption(f"Search rank: {source.rank}")


def show_result_error(error: str | None) -> bool:
    """Display expected backend failures without exposing Python tracebacks."""
    if error:
        st.error(error)
        return True
    return False


def render_sidebar() -> tuple[str, Any, Any]:
    """Render workspace selection and explicit multi-file processing controls."""
    with st.sidebar:
        st.header("Workspace")
        workspace_id = st.text_input(
            "Workspace name",
            value=st.session_state.get("active_workspace", "default"),
            help="Use letters, numbers, underscores, or hyphens.",
        ).strip()

        if not workspace_id:
            st.error("Enter a workspace name to continue.")
            st.stop()

        try:
            embedding_model, vector_store = get_workspace_resources(workspace_id)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        initialize_session_state(workspace_id)
        try:
            chunk_count = get_workspace_document_count(vector_store)
            st.caption(f"Indexed chunks: {chunk_count}")
        except Exception:
            st.caption("Workspace information is unavailable until Chroma is reachable.")

        st.divider()
        st.header("Document upload")
        uploads = st.file_uploader(
            "PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if uploads:
            st.caption(f"Selected files: {len(uploads)}")

        if st.button("Process Documents", type="primary", use_container_width=True):
            if not uploads:
                st.warning("Select at least one PDF or TXT file first.")
            else:
                uploaded_documents = [
                    UploadedDocumentData(upload.name, upload.getvalue()) for upload in uploads
                ]
                with st.spinner("Loading, chunking, embedding, and indexing documents..."):
                    result = process_uploaded_documents(
                        uploaded_documents,
                        vector_store,
                        workspace_id,
                    )
                st.session_state.last_processing = result
                if result.processed_files:
                    st.success(
                        f"Processed {len(result.processed_files)} file(s); "
                        f"indexed {result.chunks_indexed} chunk(s)."
                    )
                for error in result.errors:
                    st.error(error)

        return workspace_id, embedding_model, vector_store


def render_ask_documents(embedding_model: Any, vector_store: Any, generation_model: Any) -> None:
    """Render the existing document RAG capability."""
    st.subheader("Ask Documents")
    question = st.text_input("Question about this workspace", key="ask_question")
    if st.button("Ask documents", key="ask_button"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif get_workspace_document_count(vector_store) == 0:
            st.warning("Process at least one document in this workspace first.")
        else:
            with st.spinner("Retrieving document evidence and generating an answer..."):
                st.session_state.ask_result = answer_question(
                    question,
                    vector_store,
                    embedding_model,
                    generation_model=generation_model,
                )

    result = st.session_state.get("ask_result")
    if result:
        if not show_result_error(result.error):
            st.subheader("Answer")
            st.write(result.answer)
            show_document_sources(result.sources, result.source_documents)


def render_web_research(generation_model: Any) -> None:
    """Render the existing one-step web research capability."""
    st.subheader("Web Research")
    question = st.text_input("Current or external research question", key="web_question")
    if st.button("Research the web", key="web_button"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching the web and generating a grounded answer..."):
                st.session_state.web_result = answer_web_question(
                    question,
                    generation_model=generation_model,
                )

    result: WebResearchResult | None = st.session_state.get("web_result")
    if result:
        if not show_result_error(result.error):
            st.subheader("Answer")
            st.write(result.answer)
            show_web_sources(result.web_sources)


def render_deep_research(embedding_model: Any, vector_store: Any, generation_model: Any) -> None:
    """Render the existing bounded LangGraph Deep Research workflow."""
    st.subheader("Deep Research")
    question = st.text_area("Research question", key="deep_question", height=100)
    if st.button("Run deep research", key="deep_button"):
        if not question.strip():
            st.warning("Enter a research question first.")
        else:
            routed_question = question
            if not any(word in question.lower() for word in ("research", "approaches", "challenges", "compare")):
                routed_question = f"Research: {question}"
            with st.spinner("Planning searches, collecting evidence, and synthesizing research..."):
                st.session_state.deep_result = invoke_document_workflow(
                    routed_question,
                    vector_store,
                    embedding_model,
                    generation_model=generation_model,
                )

    result = st.session_state.get("deep_result")
    if result:
        if show_result_error(result.get("error")):
            return
        st.subheader("Research Plan")
        for index, query in enumerate(result.get("research_queries", []), start=1):
            st.write(f"{index}. {query}")
        st.subheader("Final Synthesis")
        st.write(result.get("answer", ""))
        show_web_sources(result.get("web_sources", []))
        if result.get("web_evidence"):
            with st.expander("View collected evidence"):
                for evidence in result["web_evidence"]:
                    st.markdown(format_web_source(evidence.source))
                    st.caption(f"Found for: {evidence.research_query}")
                    st.write(evidence.source.content)


def render_study_mode(embedding_model: Any, vector_store: Any, generation_model: Any) -> None:
    """Render direct calls to the existing StudyMode service."""
    st.subheader("Study Mode")
    if get_workspace_document_count(vector_store) == 0:
        st.info("Process documents in this workspace before generating study material.")

    study_mode = StudyMode(vector_store, embedding_model, generation_model=generation_model)
    summary_tab, quiz_tab, flashcard_tab = st.tabs(["Summary", "Quiz", "Flashcards"])

    with summary_tab:
        topic = st.text_input("Topic to summarize", key="summary_topic")
        if st.button("Generate summary", key="summary_button"):
            with st.spinner("Retrieving study material and generating a summary..."):
                st.session_state.study_summary_result = study_mode.summarize(topic)
        result: TopicSummaryResult | None = st.session_state.get("study_summary_result")
        if result:
            if not show_result_error(result.error):
                st.write(result.summary)
                show_document_sources(result.sources, result.source_documents)

    with quiz_tab:
        topic = st.text_input("Quiz topic", key="quiz_topic")
        count = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=get_study_quiz_max_questions(),
            value=min(3, get_study_quiz_max_questions()),
            key="quiz_count",
        )
        if st.button("Generate quiz", key="quiz_button"):
            with st.spinner("Retrieving study material and generating quiz questions..."):
                st.session_state.study_quiz_result = study_mode.generate_quiz(topic, int(count))
                for key in list(st.session_state):
                    if key.startswith("quiz_reveal_"):
                        del st.session_state[key]
        result: QuizResult | None = st.session_state.get("study_quiz_result")
        if result:
            if not show_result_error(result.error):
                for index, question in enumerate(result.questions, start=1):
                    st.markdown(f"**{index}. {question.question}**")
                    st.radio("Choose an answer", question.options, key=f"quiz_choice_{index}")
                    if st.button("Reveal answer", key=f"quiz_reveal_button_{index}"):
                        st.session_state[f"quiz_reveal_{index}"] = True
                    if st.session_state.get(f"quiz_reveal_{index}"):
                        st.success(f"Correct answer: {question.correct_answer}")
                        st.caption(question.explanation)
                show_document_sources(result.sources, result.source_documents)

    with flashcard_tab:
        topic = st.text_input("Flashcard topic", key="flashcard_topic")
        count = st.number_input(
            "Number of flashcards",
            min_value=1,
            max_value=get_study_flashcard_max_count(),
            value=min(3, get_study_flashcard_max_count()),
            key="flashcard_count",
        )
        if st.button("Generate flashcards", key="flashcard_button"):
            with st.spinner("Retrieving study material and generating flashcards..."):
                st.session_state.study_flashcard_result = study_mode.generate_flashcards(topic, int(count))
        result: FlashcardResult | None = st.session_state.get("study_flashcard_result")
        if result:
            if not show_result_error(result.error):
                for index, flashcard in enumerate(result.flashcards, start=1):
                    with st.container(border=True):
                        st.markdown(f"**{index}. {flashcard.front}**")
                        if st.button("Reveal back", key=f"flashcard_reveal_button_{index}"):
                            st.session_state[f"flashcard_reveal_{index}"] = True
                        if st.session_state.get(f"flashcard_reveal_{index}"):
                            st.write(flashcard.back)
                show_document_sources(result.sources, result.source_documents)


def main() -> None:
    """Run the thin Streamlit integration layer over the existing backend."""
    st.title("AI Research & Study Copilot")
    st.caption("Ask uploaded documents, research the web, and create grounded study material.")

    _workspace_id, embedding_model, vector_store = render_sidebar()
    try:
        generation_model = get_cached_generation_model()
    except Exception as error:
        st.error(f"Could not initialize the local generation model: {error}")
        st.stop()

    ask_tab, web_tab, deep_tab, study_tab = st.tabs(
        ["Ask Documents", "Web Research", "Deep Research", "Study Mode"]
    )
    with ask_tab:
        render_ask_documents(embedding_model, vector_store, generation_model)
    with web_tab:
        render_web_research(generation_model)
    with deep_tab:
        render_deep_research(embedding_model, vector_store, generation_model)
    with study_tab:
        render_study_mode(embedding_model, vector_store, generation_model)


if __name__ == "__main__":
    main()
