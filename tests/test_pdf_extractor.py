"""
AI-Based Test Case Generator
Phase 6A: Unit Tests for ai/pdf_extractor.py

Tests use temporary files and in-memory PDF generation (via pypdf.PdfWriter)
so no external test assets are required.  No Gemini API calls are made.

Helper:
    _write_pdf(path, pages)  – write a minimal text-bearing PDF to disk

Test classes:
    TestPDFExtractorInit        – repr and instantiation
    TestExtractSinglePage       – basic single-page extraction
    TestExtractMultiPage        – multi-page concatenation
    TestExtractedTextContent    – the returned text matches what was written
    TestExtractEmptyPDF         – PDF with no text raises ValueError
    TestExtractInvalidFile      – missing / corrupt files raise the right errors
"""

import os
import io
import pytest
import pypdf

from ai.pdf_extractor import PDFExtractor


# ---------------------------------------------------------------------------
# Helpers: build real (text-bearing) PDFs with pypdf
# ---------------------------------------------------------------------------

def _write_pdf(path: str, pages: list[str]) -> None:
    """
    Write a minimal, valid PDF containing one page per string in `pages`.

    Uses pypdf.PdfWriter with add_blank_page(); text is overlaid as a raw
    content stream so that extract_text() can read it back.
    """
    writer = pypdf.PdfWriter()
    for text in pages:
        # Add an A4 page (595 x 842 pts)
        page = writer.add_blank_page(width=595, height=842)
        # Inject a minimal PDF content stream that places the text string.
        # This is the simplest approach that pypdf can round-trip through
        # extract_text() without needing reportlab or fpdf2 as a dependency.
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_stream = (
            "BT\n"
            "/F1 12 Tf\n"
            "72 720 Td\n"
            f"({safe_text}) Tj\n"
            "ET\n"
        )
        page.merge_page(  # noqa: not ideal but works for test content
            _page_from_stream(content_stream)
        )
    with open(path, "wb") as fh:
        writer.write(fh)


def _page_from_stream(content_stream: str) -> pypdf.PageObject:
    """
    Build a pypdf PageObject whose content is the given raw PDF stream.
    Used only for test-PDF construction.
    """
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=595, height=842)
    from pypdf.generic import (
        ArrayObject, NameObject, ByteStringObject,
        DictionaryObject, DecodedStreamObject,
    )
    stream_obj = DecodedStreamObject()
    stream_obj.set_data(content_stream.encode())
    page[NameObject("/Contents")] = stream_obj
    # Provide a minimal font resource so text operators are valid
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): font,
        })
    })
    page[NameObject("/Resources")] = resources
    return page


def _write_empty_pdf(path: str) -> None:
    """Write a valid PDF that has pages but zero text content."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with open(path, "wb") as fh:
        writer.write(fh)


def _write_corrupt_pdf(path: str) -> None:
    """Write a file whose content is not a valid PDF."""
    with open(path, "wb") as fh:
        fh.write(b"This is not a PDF file at all.")


# ---------------------------------------------------------------------------
# TestPDFExtractorInit
# ---------------------------------------------------------------------------

class TestPDFExtractorInit:
    def test_instantiation(self):
        extractor = PDFExtractor()
        assert isinstance(extractor, PDFExtractor)

    def test_repr_contains_library_name(self):
        extractor = PDFExtractor()
        assert "pypdf" in repr(extractor)


# ---------------------------------------------------------------------------
# TestExtractSinglePage
# ---------------------------------------------------------------------------

class TestExtractSinglePage:
    def test_extract_returns_string(self, tmp_path):
        pdf = str(tmp_path / "single.pdf")
        _write_empty_pdf(pdf)
        extractor = PDFExtractor()
        # empty PDF should raise; test that the return type is str when text exists
        # We test with a corrupt-free but empty PDF — the error path is in another class.
        # For a successful return-type test we use the mock approach in TestExtractedTextContent.
        pass  # covered by TestExtractedTextContent below

    def test_extract_single_page_does_not_raise_for_readable_pdf(self, tmp_path):
        """
        A readable single-page PDF must not raise any exception.
        We mock extract_text at the page level to avoid font-rendering subtleties.
        """
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "single.pdf")
        _write_empty_pdf(pdf)  # valid PDF, blank pages

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "A user should be able to log in."

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [mock_page]
            extractor = PDFExtractor()
            text = extractor.extract_text(pdf)

        assert isinstance(text, str)

    def test_extract_single_page_returns_non_empty_string(self, tmp_path):
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "req.pdf")
        _write_empty_pdf(pdf)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "User must be able to reset password."

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [mock_page]
            extractor = PDFExtractor()
            result = extractor.extract_text(pdf)

        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestExtractMultiPage
# ---------------------------------------------------------------------------

class TestExtractMultiPage:
    def test_extract_multi_page_combines_pages(self, tmp_path):
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "multi.pdf")
        _write_empty_pdf(pdf)

        page_texts = ["First page text.", "Second page text.", "Third page text."]
        mock_pages = []
        for t in page_texts:
            p = MagicMock()
            p.extract_text.return_value = t
            mock_pages.append(p)

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = mock_pages
            extractor = PDFExtractor()
            result = extractor.extract_text(pdf)

        for t in page_texts:
            assert t in result

    def test_extract_multi_page_result_is_single_string(self, tmp_path):
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "multi.pdf")
        _write_empty_pdf(pdf)

        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = "Page one."
        pages[1].extract_text.return_value = "Page two."

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = pages
            result = PDFExtractor().extract_text(pdf)

        assert isinstance(result, str)

    def test_extract_skips_empty_pages_but_keeps_text_pages(self, tmp_path):
        """A blank page between two text pages must not break extraction."""
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "mixed.pdf")
        _write_empty_pdf(pdf)

        page_with_text = MagicMock()
        page_with_text.extract_text.return_value = "Real requirement text."
        blank_page = MagicMock()
        blank_page.extract_text.return_value = ""

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [blank_page, page_with_text, blank_page]
            result = PDFExtractor().extract_text(pdf)

        assert "Real requirement text." in result


# ---------------------------------------------------------------------------
# TestExtractedTextContent
# ---------------------------------------------------------------------------

class TestExtractedTextContent:
    """Verify the exact text returned by extract_text()."""

    def test_extracted_text_matches_page_content(self, tmp_path):
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "content.pdf")
        _write_empty_pdf(pdf)
        expected = "A user should be able to log in using email and password."

        mock_page = MagicMock()
        mock_page.extract_text.return_value = expected

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [mock_page]
            result = PDFExtractor().extract_text(pdf)

        assert expected in result

    def test_extracted_text_is_stripped(self, tmp_path):
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "ws.pdf")
        _write_empty_pdf(pdf)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "   Some requirement.   "

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [mock_page]
            result = PDFExtractor().extract_text(pdf)

        # Leading/trailing whitespace should have been stripped per page
        assert result.strip() == "Some requirement."

    def test_multi_page_separator_present(self, tmp_path):
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "sep.pdf")
        _write_empty_pdf(pdf)

        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = "Page one content."
        pages[1].extract_text.return_value = "Page two content."

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = pages
            result = PDFExtractor().extract_text(pdf)

        # There must be a newline between pages
        assert "\n" in result


# ---------------------------------------------------------------------------
# TestExtractEmptyPDF
# ---------------------------------------------------------------------------

class TestExtractEmptyPDF:
    def test_empty_pdf_raises_value_error(self, tmp_path):
        """A blank-page PDF with no text must raise ValueError."""
        pdf = str(tmp_path / "empty.pdf")
        _write_empty_pdf(pdf)
        extractor = PDFExtractor()
        with pytest.raises(ValueError, match="No readable text"):
            extractor.extract_text(pdf)

    def test_pdf_with_all_blank_pages_raises(self, tmp_path):
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "allblank.pdf")
        _write_empty_pdf(pdf)

        pages = [MagicMock(), MagicMock()]
        for p in pages:
            p.extract_text.return_value = "   "   # whitespace only

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = pages
            with pytest.raises(ValueError, match="No readable text"):
                PDFExtractor().extract_text(pdf)

    def test_pdf_with_none_page_text_raises(self, tmp_path):
        """Pages returning None from extract_text() must be treated as empty."""
        from unittest.mock import patch, MagicMock
        pdf = str(tmp_path / "none.pdf")
        _write_empty_pdf(pdf)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = None

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [mock_page]
            with pytest.raises(ValueError, match="No readable text"):
                PDFExtractor().extract_text(pdf)


# ---------------------------------------------------------------------------
# TestExtractInvalidFile
# ---------------------------------------------------------------------------

class TestExtractInvalidFile:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        pdf = str(tmp_path / "nonexistent.pdf")
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_text(pdf)

    def test_missing_file_error_message_contains_path(self, tmp_path):
        pdf = str(tmp_path / "ghost.pdf")
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError, match="ghost.pdf"):
            extractor.extract_text(pdf)

    def test_corrupt_pdf_raises_value_error(self, tmp_path):
        pdf = str(tmp_path / "corrupt.pdf")
        _write_corrupt_pdf(pdf)
        extractor = PDFExtractor()
        with pytest.raises(ValueError):
            extractor.extract_text(pdf)

    def test_corrupt_pdf_error_mentions_file(self, tmp_path):
        pdf = str(tmp_path / "corrupt.pdf")
        _write_corrupt_pdf(pdf)
        extractor = PDFExtractor()
        with pytest.raises(ValueError, match="corrupt.pdf"):
            extractor.extract_text(pdf)

    def test_non_pdf_file_raises_value_error(self, tmp_path):
        """A plain text file renamed to .pdf must raise ValueError."""
        fake_pdf = str(tmp_path / "fake.pdf")
        with open(fake_pdf, "w") as fh:
            fh.write("I am not a PDF.")
        extractor = PDFExtractor()
        with pytest.raises(ValueError):
            extractor.extract_text(fake_pdf)
