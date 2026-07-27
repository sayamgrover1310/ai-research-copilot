"""Tests for the document ingestion loader."""

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from src.ingestion.loader import load_document


class LoadDocumentTests(unittest.TestCase):
    def test_loads_a_text_file_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "study_notes.txt"
            file_path.write_text("Neural networks learn patterns from data.", encoding="utf-8")

            documents = load_document(file_path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "Neural networks learn patterns from data.")
        self.assertEqual(documents[0].metadata["source_filename"], "study_notes.txt")
        self.assertEqual(documents[0].metadata["file_type"], "txt")

    def test_returns_an_empty_list_for_an_empty_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "empty.txt"
            file_path.write_text("   \n", encoding="utf-8")

            documents = load_document(file_path)

        self.assertEqual(documents, [])

    def test_returns_an_empty_list_for_a_blank_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "blank.pdf"
            pdf_writer = PdfWriter()
            pdf_writer.add_blank_page(width=612, height=792)

            with file_path.open("wb") as pdf_file:
                pdf_writer.write(pdf_file)

            documents = load_document(file_path)

        self.assertEqual(documents, [])

    def test_rejects_an_unsupported_file_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "notes.docx"
            file_path.write_text("Not supported yet.", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_document(file_path)

    def test_rejects_a_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_document("does-not-exist.pdf")


if __name__ == "__main__":
    unittest.main()
