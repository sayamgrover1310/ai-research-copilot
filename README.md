# AI Research & Study Copilot

A beginner-friendly AI-assisted research and study application, built step by step.

## Current stage

Document ingestion, chunking, local embeddings, persistent local vector storage, semantic
retrieval, Basic RAG answer generation, local-document source attribution, basic web research,
bounded LangGraph Deep Research, and document-grounded Study Mode are implemented for PDF and TXT
files. A Streamlit UI and a small document-RAG evaluation framework are also implemented. No
autonomous agents have been implemented.

## Planned V1

- Create and select research workspaces
- Upload text-based PDFs
- Ask questions over uploaded documents with source citations
- Generate study notes and quizzes from workspace material

## Planned technology

- Python and Streamlit
- LangChain for document and RAG components
- Ollama with `nomic-embed-text` for local embeddings and `qwen3:4b-instruct` for local answers
- Chroma for local vector storage
- LangGraph for explicit request routing
- Tavily for current web-search evidence

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` to override local Ollama model names or storage paths. Do not
commit `.env`.

## Run the Streamlit app

Start Ollama locally, make sure the configured models are available, then run:

```bash
streamlit run app.py
```

## Project layout

```text
app.py                  Streamlit frontend for the existing backend capabilities
src/ingestion/loader.py Path-based PDF and TXT document ingestion
src/ingestion/chunker.py Splits ingested documents into overlapping text chunks
src/retrieval/embeddings.py Creates local embeddings for chunks and queries
src/retrieval/vector_store.py Persists chunks and vectors in local Chroma collections
src/retrieval/retriever.py Returns relevant evidence chunks from Chroma
src/rag/basic_rag.py Retrieves evidence and generates a grounded answer with Ollama
src/rag/citations.py Creates deterministic source records from retrieved Documents
src/workflow/document_workflow.py Routes questions to document RAG or general chat
src/web_research/tavily_search.py Tavily-specific search and result normalization
src/web_research/basic_web_research.py Builds grounded answers from one web search
src/web_research/deep_research.py Plans, searches, deduplicates, and synthesizes bounded research
src/study/study_mode.py Creates document-grounded summaries, quizzes, and flashcards
src/services/workspace_service.py Reuses ingestion, chunking, and storage for UI uploads
src/                    Application modules, added one feature at a time
data/uploads/           Local uploaded files (not committed)
data/chroma/            Local vector database files (not committed)
tests/                  Tests, added alongside implemented features
```
