import pytest

from ai.csv_extractor import CSVExtractor


def create_csv(path, content):
    path.write_text(content, encoding="utf-8")


def test_single_row(tmp_path):
    file_path = tmp_path / "requirement.csv"

    create_csv(
        file_path,
        "Requirement\nUser should be able to login."
    )

    extractor = CSVExtractor()

    result = extractor.extract_text(file_path)

    assert "Requirement" in result
    assert "User should be able to login." in result


def test_multiple_rows(tmp_path):
    file_path = tmp_path / "requirements.csv"

    create_csv(
        file_path,
        "Requirement\n"
        "User enters username.\n"
        "User enters password.\n"
        "System authenticates the user."
    )

    extractor = CSVExtractor()

    result = extractor.extract_text(file_path)

    assert "User enters username." in result
    assert "User enters password." in result
    assert "System authenticates the user." in result


def test_multiple_columns(tmp_path):
    file_path = tmp_path / "requirements.csv"

    create_csv(
        file_path,
        "Actor,Action,Expected Result\n"
        "User,Login,Dashboard is displayed"
    )

    extractor = CSVExtractor()

    result = extractor.extract_text(file_path)

    assert "User" in result
    assert "Login" in result
    assert "Dashboard is displayed" in result


def test_missing_file(tmp_path):
    file_path = tmp_path / "missing.csv"

    extractor = CSVExtractor()

    with pytest.raises(FileNotFoundError):
        extractor.extract_text(file_path)


def test_empty_csv(tmp_path):
    file_path = tmp_path / "empty.csv"

    create_csv(file_path, "")

    extractor = CSVExtractor()

    with pytest.raises(ValueError):
        extractor.extract_text(file_path)
