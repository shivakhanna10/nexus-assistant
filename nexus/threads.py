"""Conversation registry: titles, owners, timestamps and attached documents.

Stored in the LangGraph store (the same backend as long-term memory), so it
lives in SQLite locally and in Postgres when ``POSTGRES_URI`` is set, which is
what makes conversations survive restarts on hosts with ephemeral disks
(e.g. Streamlit Community Cloud).

    ("threads",)               key=thread_id   {user_id, title, created_at, updated_at}
    ("thread_docs", thread_id) key=index_key   {filename, pages, chunks, added_at}
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass

from .config import settings
from .persistence import get_store

log = logging.getLogger(__name__)

THREADS_NS = ("threads",)
LIST_LIMIT = 1000


def _docs_ns(thread_id: str) -> tuple[str, ...]:
    return ("thread_docs", thread_id)


@dataclass
class ThreadInfo:
    thread_id: str
    user_id: str
    title: str
    created_at: float
    updated_at: float


def new_thread_id() -> str:
    return str(uuid.uuid4())


def _info(thread_id: str, v: dict) -> ThreadInfo:
    return ThreadInfo(thread_id, v.get("user_id", "default"), v.get("title", "New chat"),
                      float(v.get("created_at", 0)), float(v.get("updated_at", 0)))


def get_thread(thread_id: str) -> ThreadInfo | None:
    item = get_store().get(THREADS_NS, thread_id)
    return _info(thread_id, item.value) if item else None


def upsert_thread(thread_id: str, user_id: str, title: str | None = None) -> None:
    now = time.time()
    current = get_store().get(THREADS_NS, thread_id)
    value = dict(current.value) if current else {"user_id": user_id, "title": "New chat", "created_at": now}
    value["updated_at"] = now
    if title:
        value["title"] = title
    get_store().put(THREADS_NS, thread_id, value, index=False)


def rename_thread(thread_id: str, title: str) -> None:
    current = get_store().get(THREADS_NS, thread_id)
    if current:
        get_store().put(THREADS_NS, thread_id, {**current.value, "title": title.strip()[:80]}, index=False)


def list_threads(user_id: str) -> list[ThreadInfo]:
    items = get_store().search(THREADS_NS, filter={"user_id": user_id}, limit=LIST_LIMIT)
    return sorted((_info(it.key, it.value) for it in items), key=lambda t: t.updated_at, reverse=True)


def delete_thread(thread_id: str) -> None:
    store = get_store()
    store.delete(THREADS_NS, thread_id)
    for it in store.search(_docs_ns(thread_id), limit=LIST_LIMIT):
        store.delete(_docs_ns(thread_id), it.key)


def attach_document(thread_id: str, index_key: str, filename: str, pages: int, chunks: int) -> None:
    get_store().put(_docs_ns(thread_id), index_key,
                    {"filename": filename, "pages": pages, "chunks": chunks, "added_at": time.time()},
                    index=False)


def detach_document(thread_id: str, index_key: str) -> None:
    get_store().delete(_docs_ns(thread_id), index_key)


def thread_documents(thread_id: str) -> list[dict]:
    """Documents attached to a chat whose index is still on disk (on hosts with an
    ephemeral disk, indexes vanish on restart and the file must be uploaded again)."""
    from .rag.indexing import index_exists  # local import: indexing imports this module

    items = get_store().search(_docs_ns(thread_id), limit=LIST_LIMIT)
    docs = [{"thread_id": thread_id, "index_key": it.key, **it.value} for it in items]
    return sorted((d for d in docs if index_exists(d["index_key"])), key=lambda d: d.get("added_at", 0))


# ------------------------------------------------- one-time legacy migration
def _migrate_legacy_registry() -> None:
    """Import conversations from the old registry.db (pre-store versions), once."""
    legacy = settings.data_dir / "registry.db"
    if not legacy.exists():
        return
    try:
        conn = sqlite3.connect(str(legacy))
        conn.row_factory = sqlite3.Row
        store = get_store()
        for r in conn.execute("SELECT * FROM threads").fetchall():
            if store.get(THREADS_NS, r["thread_id"]) is None:
                store.put(THREADS_NS, r["thread_id"], {
                    "user_id": r["user_id"], "title": r["title"],
                    "created_at": r["created_at"], "updated_at": r["updated_at"]}, index=False)
        for r in conn.execute("SELECT * FROM thread_docs").fetchall():
            attach_document(r["thread_id"], r["index_key"], r["filename"], r["pages"], r["chunks"])
        conn.close()
        legacy.rename(legacy.with_suffix(".db.migrated"))
        log.info("Migrated legacy conversation registry into the store.")
    except Exception as exc:
        log.warning("Could not migrate legacy registry: %s", exc)


_migrate_legacy_registry()
