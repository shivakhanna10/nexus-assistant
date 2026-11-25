"""Persistence layer: short-term memory (checkpointer) + long-term memory (store).

* SQLite by default (zero setup) – from ``chatbot-in-langgraph`` (SqliteSaver).
* Postgres when ``POSTGRES_URI`` is set – from ``memory-in-llms`` (PostgresSaver /
  PostgresStore). ``docker compose up -d`` starts a matching database.
"""
from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from .config import settings
from .llm import embedding_dims, get_embeddings

log = logging.getLogger(__name__)


def _index_config() -> dict | None:
    if not settings.semantic_memory:
        return None
    return {"embed": get_embeddings(), "dims": embedding_dims(), "fields": ["data"]}


@lru_cache(maxsize=1)
def _sqlite_conn() -> sqlite3.Connection:
    """Checkpointer connection (SqliteSaver serialises access with its own lock)."""
    conn = sqlite3.connect(str(settings.sqlite_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@lru_cache(maxsize=1)
def _sqlite_store_conn() -> sqlite3.Connection:
    """Separate autocommit connection: SqliteStore issues its own BEGIN/COMMIT, and the
    memory node runs in parallel with the agent, so it must not share the saver's connection."""
    conn = sqlite3.connect(str(settings.data_dir / "memory.db"), check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@lru_cache(maxsize=1)
def _pg_pool():
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    return ConnectionPool(
        settings.postgres_uri,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )


@lru_cache(maxsize=1)
def get_checkpointer() -> BaseCheckpointSaver:
    if settings.postgres_uri:
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = PostgresSaver(_pg_pool())
        saver.setup()
        return saver

    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(_sqlite_conn())
    saver.setup()
    return saver


@lru_cache(maxsize=1)
def get_store() -> BaseStore:
    index = _index_config()
    if settings.postgres_uri:
        from langgraph.store.postgres import PostgresStore

        store = PostgresStore(_pg_pool(), index=index) if index else PostgresStore(_pg_pool())
        store.setup()
        return store

    try:
        from langgraph.store.sqlite import SqliteStore

        conn = _sqlite_store_conn()
        store = SqliteStore(conn, index=index) if index else SqliteStore(conn)
        store.setup()
        return store
    except Exception as exc:  # pragma: no cover - depends on installed extras
        log.warning("SqliteStore unavailable (%s); long-term memory will not survive restarts.", exc)
        from langgraph.store.memory import InMemoryStore

        return InMemoryStore(index=index) if index else InMemoryStore()


def backend_name() -> str:
    return "Postgres" if settings.postgres_uri else "SQLite"
