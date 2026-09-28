"""
AI-Based Test Case Generator
Phase 5B: Unit Tests for database/db.py

Every test uses an isolated in-memory (or temporary file) SQLite database so
that the real data/test_cases.db is never touched.  No Gemini API calls are made.

Test classes:
    TestDatabaseInitialization  – initialize_database() creates the table
    TestSaveGeneration          – save_generation() inserts rows correctly
    TestGetGenerations          – get_generations() returns rows newest-first
    TestGetGeneration           – get_generation() fetches by id / returns None
    TestJsonPreservation        – structured JSON columns round-trip correctly
    TestMultipleGenerations     – ordering and isolation with several rows
"""

import os
import json
import sqlite3
import tempfile
import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-key")

from database.db import (
    initialize_database,
    save_generation,
    get_generations,
    get_generation,
    _get_connection,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tmp_db(tmp_path) -> str:
    """Return a path to a fresh temporary database file."""
    return str(tmp_path / "test.db")


def _sample_processed_req() -> dict:
    return {
        "actor": "registered user",
        "action": "log in",
        "conditions": ["valid account exists"],
        "inputs": ["email", "password"],
        "expected_outcome": "user is authenticated and redirected",
    }


def _sample_scenarios() -> dict:
    return {
        "positive_scenarios": ["Login succeeds with valid credentials."],
        "negative_scenarios": ["Login fails with wrong password."],
        "boundary_scenarios": [],
        "edge_cases": ["SQL injection in email field."],
    }


def _sample_test_cases() -> list:
    return [
        {
            "title": "TC-001: Valid Login",
            "description": "Verify login with correct credentials.",
            "steps": ["Go to /login", "Enter email", "Click Submit"],
            "expected_result": "User lands on dashboard.",
            "priority": "High",
        }
    ]


def _save_one(db_path: str, requirement: str = "A user should be able to log in.") -> int:
    """Insert one sample generation and return its id."""
    return save_generation(
        requirement=requirement,
        processed_requirement=_sample_processed_req(),
        scenarios=_sample_scenarios(),
        test_cases=_sample_test_cases(),
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# TestDatabaseInitialization
# ---------------------------------------------------------------------------

class TestDatabaseInitialization:
    def test_initialize_creates_db_file(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        assert os.path.exists(db_path)

    def test_initialize_creates_generations_table(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='generations'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None, "generations table was not created"

    def test_initialize_is_idempotent(self, tmp_path):
        """Calling initialize_database twice must not raise."""
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        initialize_database(db_path)   # second call – should not raise

    def test_initialize_creates_data_directory_if_missing(self, tmp_path):
        nested = tmp_path / "new_subdir" / "test.db"
        initialize_database(str(nested))
        assert nested.exists()

    def test_generations_table_has_correct_columns(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(generations)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {"id", "requirement", "processed_requirement",
                    "scenarios", "test_cases", "created_at"}
        assert expected.issubset(columns)


# ---------------------------------------------------------------------------
# TestSaveGeneration
# ---------------------------------------------------------------------------

class TestSaveGeneration:
    def test_save_returns_integer_id(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        row_id = _save_one(db_path)
        assert isinstance(row_id, int)

    def test_save_returns_positive_id(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        row_id = _save_one(db_path)
        assert row_id > 0

    def test_save_increments_id(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        id1 = _save_one(db_path, "First requirement.")
        id2 = _save_one(db_path, "Second requirement.")
        assert id2 == id1 + 1

    def test_save_persists_requirement_text(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        req = "A user should be able to reset their password."
        row_id = _save_one(db_path, req)
        row = get_generation(row_id, db_path)
        assert row["requirement"] == req

    def test_save_persists_created_at(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        row_id = _save_one(db_path)
        row = get_generation(row_id, db_path)
        assert row["created_at"] is not None
        assert len(row["created_at"]) > 0


# ---------------------------------------------------------------------------
# TestGetGenerations
# ---------------------------------------------------------------------------

class TestGetGenerations:
    def test_get_generations_returns_list(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        result = get_generations(db_path)
        assert isinstance(result, list)

    def test_get_generations_empty_when_no_rows(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        result = get_generations(db_path)
        assert result == []

    def test_get_generations_returns_all_rows(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        _save_one(db_path, "First.")
        _save_one(db_path, "Second.")
        _save_one(db_path, "Third.")
        result = get_generations(db_path)
        assert len(result) == 3

    def test_get_generations_ordered_newest_first(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        id1 = _save_one(db_path, "Oldest requirement.")
        id2 = _save_one(db_path, "Newest requirement.")
        rows = get_generations(db_path)
        # Newest (highest id) should come first
        assert rows[0]["id"] == id2
        assert rows[1]["id"] == id1

    def test_get_generations_each_row_is_dict(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        _save_one(db_path)
        rows = get_generations(db_path)
        assert isinstance(rows[0], dict)


# ---------------------------------------------------------------------------
# TestGetGeneration
# ---------------------------------------------------------------------------

class TestGetGeneration:
    def test_get_generation_returns_dict(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        row_id = _save_one(db_path)
        row = get_generation(row_id, db_path)
        assert isinstance(row, dict)

    def test_get_generation_correct_id(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        row_id = _save_one(db_path)
        row = get_generation(row_id, db_path)
        assert row["id"] == row_id

    def test_get_generation_nonexistent_returns_none(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        result = get_generation(9999, db_path)
        assert result is None

    def test_get_generation_returns_correct_requirement(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        req = "User should be able to change their email address."
        row_id = save_generation(
            requirement=req,
            processed_requirement=_sample_processed_req(),
            scenarios=_sample_scenarios(),
            test_cases=_sample_test_cases(),
            db_path=db_path,
        )
        row = get_generation(row_id, db_path)
        assert row["requirement"] == req

    def test_get_generation_fetches_specific_row(self, tmp_path):
        """With multiple rows, get_generation returns only the requested one."""
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        id1 = _save_one(db_path, "First requirement.")
        _save_one(db_path, "Second requirement.")
        row = get_generation(id1, db_path)
        assert row["requirement"] == "First requirement."


# ---------------------------------------------------------------------------
# TestJsonPreservation
# ---------------------------------------------------------------------------

class TestJsonPreservation:
    def test_processed_requirement_round_trips(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        pr = _sample_processed_req()
        row_id = save_generation(
            requirement="req",
            processed_requirement=pr,
            scenarios=_sample_scenarios(),
            test_cases=_sample_test_cases(),
            db_path=db_path,
        )
        row = get_generation(row_id, db_path)
        assert row["processed_requirement"] == pr

    def test_scenarios_round_trips(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        sc = _sample_scenarios()
        row_id = save_generation(
            requirement="req",
            processed_requirement=_sample_processed_req(),
            scenarios=sc,
            test_cases=_sample_test_cases(),
            db_path=db_path,
        )
        row = get_generation(row_id, db_path)
        assert row["scenarios"] == sc

    def test_test_cases_round_trips(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        tcs = _sample_test_cases()
        row_id = save_generation(
            requirement="req",
            processed_requirement=_sample_processed_req(),
            scenarios=_sample_scenarios(),
            test_cases=tcs,
            db_path=db_path,
        )
        row = get_generation(row_id, db_path)
        assert row["test_cases"] == tcs

    def test_processed_requirement_is_dict_not_string(self, tmp_path):
        """Structured columns must be returned as Python objects, not raw JSON strings."""
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        row_id = _save_one(db_path)
        row = get_generation(row_id, db_path)
        assert isinstance(row["processed_requirement"], dict)

    def test_scenarios_is_dict_not_string(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        row_id = _save_one(db_path)
        row = get_generation(row_id, db_path)
        assert isinstance(row["scenarios"], dict)

    def test_test_cases_is_list_not_string(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        row_id = _save_one(db_path)
        row = get_generation(row_id, db_path)
        assert isinstance(row["test_cases"], list)

    def test_test_cases_list_has_expected_fields(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        row_id = _save_one(db_path)
        row = get_generation(row_id, db_path)
        tc = row["test_cases"][0]
        for field in ("title", "description", "steps", "expected_result", "priority"):
            assert field in tc


# ---------------------------------------------------------------------------
# TestMultipleGenerations
# ---------------------------------------------------------------------------

class TestMultipleGenerations:
    def test_multiple_saves_produce_unique_ids(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        ids = [_save_one(db_path, f"Requirement {i}.") for i in range(5)]
        assert len(set(ids)) == 5

    def test_get_generations_count_matches_saves(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        for i in range(4):
            _save_one(db_path, f"Requirement {i}.")
        rows = get_generations(db_path)
        assert len(rows) == 4

    def test_each_generation_has_its_own_requirement(self, tmp_path):
        db_path = _tmp_db(tmp_path)
        initialize_database(db_path)
        requirements = [f"Requirement number {i}." for i in range(3)]
        ids = [_save_one(db_path, r) for r in requirements]
        for row_id, req in zip(ids, requirements):
            row = get_generation(row_id, db_path)
            assert row["requirement"] == req
