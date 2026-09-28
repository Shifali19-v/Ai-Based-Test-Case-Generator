"""
AI-Based Test Case Generator
Phase 6A: PDF Text Extractor

Provides the PDFExtractor class, which extracts plain text from a text-based
PDF file using the `pypdf` library.

This phase supports text-based PDFs only.
OCR and image/vision processing are NOT included.

Usage:
    extractor = PDFExtractor()
    text = extractor.extract_text("/path/to/requirement.pdf")
    # `text` is a single string containing all readable page content.
"""

from pathlib import Path
import pypdf


class PDFExtractor:
    """
    Extracts plain text from a text-based PDF file.

    Each page's text is extracted and joined with a newline so that the
    output is one continuous string ready to be fed into the pipeline.

    Only text-based PDFs are supported.  Scanned PDFs (image-only) will
    raise a ValueError because no text can be extracted.
    """

    # Separator inserted between pages so content from different pages is
    # clearly delimited (useful if the requirement spans multiple pages).
    PAGE_SEPARATOR = "\n"

    def extract_text(self, file_path: str) -> str:
        """
        Extract and return the full text content of a PDF file.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            A single string containing the text from all readable pages,
            with pages separated by a newline.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError:        If the file cannot be read as a PDF, or if
                               no readable text is found in the document.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as exc:
            raise ValueError(
                f"Could not read PDF file '{file_path}': {exc}"
            ) from exc

        page_texts: list[str] = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                # A single unreadable page is skipped; we collect what we can.
                page_text = ""

            stripped = page_text.strip()
            if stripped:
                page_texts.append(stripped)

        if not page_texts:
            raise ValueError(
                "No readable text found in the PDF. "
                "The file may be empty, image-only, or corrupted."
            )

        return self.PAGE_SEPARATOR.join(page_texts)

    def __repr__(self) -> str:
        return "PDFExtractor(library='pypdf')"
