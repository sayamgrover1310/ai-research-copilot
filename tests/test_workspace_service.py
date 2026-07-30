"""Unit tests for the UI-facing workspace indexing service."""

import tempfile
import unittest
from pathlib import Path

from src.retrieval.vector_store import get_vector_store
from src.services.workspace_service import (
    UploadedDocumentData,
    get_workspace_document_count,
    process_uploaded_documents,
)


class FakeEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class WorkspaceServiceTests(unittest.TestCase):
    def test_processes_uploaded_text_into_the_selected_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            vector_store = get_vector_store(
                embedding_model=FakeEmbeddingModel(),
                persist_directory=temporary_directory,
                workspace_id="study_workspace",
            )
            result = process_uploaded_documents(
                [UploadedDocumentData("rag_notes.txt", b"RAG retrieves external knowledge.")],
                vector_store,
                "study_workspace",
                uploads_directory=Path(temporary_directory) / "uploads",
            )

            self.assertEqual(result.processed_files, ["rag_notes.txt"])
            self.assertEqual(result.errors, [])
            self.assertEqual(result.chunks_indexed, 1)
            self.assertEqual(get_workspace_document_count(vector_store), 1)

    def test_reports_invalid_uploaded_files_without_stopping_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            vector_store = get_vector_store(
                embedding_model=FakeEmbeddingModel(),
                persist_directory=temporary_directory,
                workspace_id="study_workspace",
            )
            result = process_uploaded_documents(
                [
                    UploadedDocumentData("unsupported.csv", b"not supported"),
                    UploadedDocumentData("notes.txt", b"Valid text"),
                ],
                vector_store,
                "study_workspace",
                uploads_directory=Path(temporary_directory) / "uploads",
            )

            self.assertEqual(result.processed_files, ["notes.txt"])
            self.assertEqual(len(result.errors), 1)
            self.assertEqual(get_workspace_document_count(vector_store), 1)


if __name__ == "__main__":
    unittest.main()
