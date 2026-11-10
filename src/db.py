"""SQLite helpers: connection, table loading and a tiny parser for named SQL files."""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from . import config as C


@contextmanager
def connect(path: Path | None = None):
    path = Path(path or C.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def write_table(df: pd.DataFrame, name: str) -> None:
    """Create (or replace) a table in the project database."""
    with connect() as conn:
        df.to_sql(name, conn, if_exists="replace", index=False)


def query(sql: str) -> pd.DataFrame:
    """Run a SELECT against the project database and return a DataFrame."""
    with connect() as conn:
        return pd.read_sql_query(sql, conn)


def load_named_queries(path: Path) -> dict[str, tuple[str, str]]:
    """Parse a .sql file made of blocks introduced by `-- name: <id>` (and optional `-- desc: <text>`).

    Returns {name: (sql, description)}.
    """
    queries: dict[str, tuple[str, str]] = {}
    name: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if name is None:
            return
        text = "\n".join(buf).strip().rstrip(";").strip()
        desc_match = re.search(r"--\s*desc:\s*(.*)", text)
        queries[name] = (text, desc_match.group(1).strip() if desc_match else "")

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"--\s*name:\s*(\w+)", line)
        if m:
            flush()
            name, buf = m.group(1), []
        elif name is not None:
            buf.append(line)
    flush()
    return queries
