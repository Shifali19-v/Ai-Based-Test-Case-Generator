"""
AI-Based Test Case Generator
Phase 5B: SQLite Database Layer

Provides simple functions for persisting and retrieving pipeline generations
using Python's built-in sqlite3 module.  No external ORM is used.

Database file: data/test_cases.db  (created automatically if absent)

Table: generations
  id                   INTEGER PRIMARY KEY AUTOINCREMENT
  requirement          TEXT    NOT NULL
  processed_requirement TEXT    NOT NULL   (JSON string)
  scenarios            TEXT    NOT NULL   (JSON string)
  test_cases           TEXT    NOT NULL   (JSON string)
  created_at           TEXT    NOT NULL   (ISO-8601 UTC timestamp)

Public API:
    initialize_database()             – create DB file and table if needed
    save_generation(...)              – insert one row; return the new row id
    get_generations()                 – return all rows, newest first
    get_generation(generation_id)     – return one row dict, or None if missing
"""

import json
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------

# Resolve path relative to the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = _PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "test_cases.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS generations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    requirement           TEXT    NOT NULL,
    processed_requirement TEXT    NOT NULL,
    scenarios             TEXT    NOT NULL,
    test_cases            TEXT    NOT NULL,
    created_at            TEXT    NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_connection(db_path: str = str(DB_PATH)) -> sqlite3.Connection:
    """
    Open and return a sqlite3 connection with row_factory set so that
    rows are returned as dict-like objects.

    Args:
        db_path: Path to the SQLite database file.  Defaults to DB_PATH.

    Returns:
        An open sqlite3.Connection.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    """
    Convert a sqlite3.Row to a plain Python dict, deserialising JSON columns.

    Returns:
        A dict with all columns, where JSON text columns are parsed back into
        their original Python objects (dict / list).
    """
    d = dict(row)
    for col in ("processed_requirement", "scenarios", "test_cases"):
        if isinstance(d.get(col), str):
            d[col] = json.loads(d[col])
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize_database(db_path: str = str(DB_PATH)) -> None:
    """
    Create the data directory and the generations table if they do not exist.

    This is safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).

    Args:
        db_path: Path to the SQLite database file.  Defaults to DB_PATH.
    """
    # Ensure the data/ directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = _get_connection(db_path)
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def save_generation(
    requirement: str,
    processed_requirement: dict,
    scenarios: dict,
    test_cases: list,
    db_path: str = str(DB_PATH),
) -> int:
    """
    Insert one complete generation into the database.

    Structured fields (processed_requirement, scenarios, test_cases) are
    serialised to JSON strings before storage.

    Args:
        requirement:           The original plain-text requirement string.
        processed_requirement: Dict representation of ProcessedRequirement.
        scenarios:             Dict representation of ScenarioSet.
        test_cases:            List of dicts, each representing a TestCase.
        db_path:               Path to the SQLite database file.

    Returns:
        The integer id of the newly inserted row.
    """
    created_at = datetime.now(timezone.utc).isoformat()

    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO generations
                (requirement, processed_requirement, scenarios, test_cases, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                requirement,
                json.dumps(processed_requirement),
                json.dumps(scenarios),
                json.dumps(test_cases),
                created_at,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_generations(db_path: str = str(DB_PATH)) -> list[dict]:
    """
    Return all saved generations, ordered newest first.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A list of dicts; each dict contains all columns of the generations
        table, with JSON columns already parsed back to Python objects.
        Returns an empty list if the table has no rows.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM generations ORDER BY id DESC"
        )
        rows = cursor.fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def get_generation(
    generation_id: int,
    db_path: str = str(DB_PATH),
) -> Optional[dict]:
    """
    Return one generation by its primary key, or None if it does not exist.

    Args:
        generation_id: The integer id of the generation to fetch.
        db_path:       Path to the SQLite database file.

    Returns:
        A dict of the row with JSON columns parsed, or None.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM generations WHERE id = ?",
            (generation_id,),
        )
        row = cursor.fetchone()
        return _row_to_dict(row) if row is not None else None
    finally:
        conn.close()
