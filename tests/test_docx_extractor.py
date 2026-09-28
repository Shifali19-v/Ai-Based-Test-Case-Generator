from docx import Document
import pytest

from ai.docx_extractor import DOCXExtractor


def create_docx(path, paragraphs):
    document = Document()

    for paragraph in paragraphs:
        document.add_paragraph(paragraph)

    document.save(path)


def test_single_paragraph(tmp_path):
    file_path = tmp_path / "requirement.docx"

    create_docx(file_path, ["User should be able to login."])

    extractor = DOCXExtractor()

    result = extractor.extract_text(file_path)

    assert result == "User should be able to login."


def test_multiple_paragraphs(tmp_path):
    file_path = tmp_path / "requirement.docx"

    create_docx(
        file_path,
        [
            "User enters username.",
            "User enters password.",
            "System authenticates the user."
        ]
    )

    extractor = DOCXExtractor()

    result = extractor.extract_text(file_path)

    assert "User enters username." in result
    assert "User enters password." in result
    assert "System authenticates the user." in result


def test_missing_file(tmp_path):
    file_path = tmp_path / "missing.docx"

    extractor = DOCXExtractor()

    with pytest.raises(FileNotFoundError):
        extractor.extract_text(file_path)


def test_empty_docx(tmp_path):
    file_path = tmp_path / "empty.docx"

    create_docx(file_path, [])

    extractor = DOCXExtractor()

    with pytest.raises(ValueError):
        extractor.extract_text(file_path)


def test_corrupt_docx(tmp_path):
    file_path = tmp_path / "corrupt.docx"

    file_path.write_text("This is not a valid DOCX file.")

    extractor = DOCXExtractor()

    with pytest.raises(ValueError):
        extractor.extract_text(file_path)
