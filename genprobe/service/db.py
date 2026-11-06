"""Database connections and schema migrations for the scoring service (D-040).

Migrations are plain SQL files in ``migrations/``, applied in name order, each once, recorded in
``schema_migrations``. A transaction-level advisory lock makes concurrent runs (two containers
starting together) wait for each other instead of applying a file twice.

    python -m genprobe.service.db migrate     # apply what is missing
    python -m genprobe.service.db status      # list applied and pending files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MIGRATIONS = Path(__file__).with_name("migrations")
_LOCK_ID = 7_402_311_040  # arbitrary, fixed: the advisory lock taken while migrating


def connect(url: str, **kwargs):
    """A psycopg connection in autocommit mode returning rows as dicts; group writes with
    ``with conn.transaction():``."""
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(url, autocommit=True, row_factory=dict_row, **kwargs)


def pool(url: str, min_size: int = 1, max_size: int = 10):
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    return ConnectionPool(url, min_size=min_size, max_size=max_size, open=False,
                          kwargs={"autocommit": True, "row_factory": dict_row})


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS.glob("*.sql"))


def _ensure_table(conn) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                 "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")


def applied(conn) -> set[str]:
    _ensure_table(conn)
    return {r["version"] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}


def pending(conn) -> list[str]:
    done = applied(conn)
    return [f.stem for f in migration_files() if f.stem not in done]


def migrate(conn) -> list[str]:
    """Apply every pending migration in one transaction; return the versions applied."""
    out = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
        done = applied(conn)
        for f in migration_files():
            if f.stem in done:
                continue
            conn.execute(f.read_text())          # no parameters: several statements are allowed
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (f.stem,))
            out.append(f.stem)
    return out


def main(argv: list[str] | None = None) -> int:
    from .config import Settings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["migrate", "status"])
    ap.add_argument("--database-url", default=None)
    a = ap.parse_args(argv)
    url = a.database_url or Settings.from_env().database_url
    with connect(url) as conn:
        if a.command == "migrate":
            done = migrate(conn)
            print("applied: " + (", ".join(done) if done else "nothing (up to date)"))
        else:
            print("applied: " + ", ".join(sorted(applied(conn))))
            print("pending: " + (", ".join(pending(conn)) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
