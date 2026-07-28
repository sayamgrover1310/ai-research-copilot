# AI Research & Study Copilot

A beginner-friendly AI-assisted research and study application, built step by step.

## Current stage

Document ingestion, chunking, local embeddings, and persistent local vector storage are
implemented for PDF and TXT files. No Streamlit UI, retrieval, RAG pipeline, LangGraph
workflow, or agents have been implemented yet.

## Planned V1

- Create and select research workspaces
- Upload text-based PDFs
- Ask questions over uploaded documents with source citations
- Generate study notes and quizzes from workspace material

## Planned technology

- Python and Streamlit
- LangChain for document and RAG components
- Ollama with `nomic-embed-text` for local embeddings
- Chroma for local vector storage
- LangGraph for explicit request routing

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

When an LLM provider is connected in a later stage, copy `.env.example` to `.env` and
add the required API key. Do not commit `.env`.

## Project layout

```text
app.py                  Future Streamlit entry point
src/ingestion/loader.py Path-based PDF and TXT document ingestion
src/ingestion/chunker.py Splits ingested documents into overlapping text chunks
src/retrieval/embeddings.py Creates local embeddings for chunks and queries
src/retrieval/vector_store.py Persists chunks and vectors in local Chroma collections
src/                    Application modules, added one feature at a time
data/uploads/           Local uploaded files (not committed)
data/chroma/            Local vector database files (not committed)
tests/                  Tests, added alongside implemented features
```
