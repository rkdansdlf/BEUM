"""Small, dependency-free migration runner for the SQLite receiver database.

The receiver is intentionally deployable without Alembic on a Raspberry Pi.
Migrations are ordered SQL files and are recorded in ``schema_migrations`` so
that an existing database is upgraded once and a fresh database follows the
same schema path.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MIGRATIONS_DIR = Path(__file__).with_name("versions")


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    path: Path


MIGRATIONS: tuple[Migration, ...] = (
    Migration("0001", "initial_schema", MIGRATIONS_DIR / "0001_initial_schema.sql"),
    Migration("0002", "add_event_metrics", MIGRATIONS_DIR / "0002_add_event_metrics.sql"),
    Migration("0003", "add_monitoring_tables", MIGRATIONS_DIR / "0003_add_monitoring_tables.sql"),
)

_ADD_EVENT_COLUMN = re.compile(
    r"^\s*ALTER\s+TABLE\s+events\s+ADD\s+COLUMN\s+"
    r"(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s+REAL\s*;?\s*$",
    re.IGNORECASE,
)


def _sql_statements(sql: str) -> Iterable[str]:
    """Yield statements without requiring a third-party SQL parser."""
    statement = ""
    for line in sql.splitlines():
        if line.lstrip().startswith("--"):
            continue
        statement += line + "\n"
        if sqlite3.complete_statement(statement):
            cleaned = statement.strip()
            if cleaned:
                yield cleaned
            statement = ""
    if statement.strip():
        raise ValueError("Migration SQL ended with an incomplete statement")


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def _apply_sql_file(conn: sqlite3.Connection, path: Path) -> None:
    for statement in _sql_statements(path.read_text(encoding="utf-8")):
        conn.execute(statement)


def _apply_event_metrics(conn: sqlite3.Connection, path: Path) -> None:
    """Apply the event-column migration safely to all legacy DB variants.

    SQLite does not support ``ADD COLUMN IF NOT EXISTS``. The migration file
    contains the canonical DDL, while this small amount of introspection makes
    it safe when an operator already added one of the columns manually.
    """
    columns = _table_columns(conn, "events")
    for statement in _sql_statements(path.read_text(encoding="utf-8")):
        match = _ADD_EVENT_COLUMN.match(statement)
        if match:
            column = match.group("column")
            if column not in columns:
                conn.execute(statement)
                columns.add(column)
            continue
        conn.execute(statement)


def _open_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def migrate_database(db_path: str | Path) -> list[str]:
    """Apply pending migrations and return the versions applied this run."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _open_connection(path)
    applied_now: list[str] = []
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )

        for migration in MIGRATIONS:
            if not migration.path.is_file():
                raise FileNotFoundError(f"Migration file not found: {migration.path}")

            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (migration.version,),
                ).fetchone()
                if row is not None:
                    conn.execute("COMMIT")
                    continue

                if migration.version == "0002":
                    _apply_event_metrics(conn, migration.path)
                else:
                    _apply_sql_file(conn, migration.path)

                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.name, datetime.now(timezone.utc).isoformat()),
                )
                conn.execute("COMMIT")
                applied_now.append(migration.version)
            except Exception:
                conn.execute("ROLLBACK")
                raise
    finally:
        conn.close()

    return applied_now


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply BEUM SQLite database migrations")
    parser.add_argument("--db", required=True, type=Path, help="Path to beum_events.db")
    args = parser.parse_args()
    applied = migrate_database(args.db)
    if applied:
        print(f"Applied migrations: {', '.join(applied)}")
    else:
        print("Database is already up to date")


if __name__ == "__main__":
    main()
