"""Document ingestion + vector indexes.

Combines:
* per-chat PDF upload & FAISS retrieval      (chatbot-in-langgraph / langraph_rag_backend)
* fingerprint-keyed on-disk index cache       (langsmith-masterclass / 3_rag_v4)
* LangSmith @traceable spans for every stage  (langsmith-masterclass / 3_rag_v2-v4)
* unicode clean-up of PDF text                (corrective-rag)
* a shared multi-file knowledge base          (corrective-rag's book corpus)
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import warnings

from langchain_core.documents import Document
from pypdf import PdfReader

with warnings.catch_warnings():
    # langchain-community is in maintenance mode; FAISS is the only thing we still use from it.
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from ..config import settings
from ..llm import get_embeddings
from .. import threads

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
INDEX_FORMAT = "v2"
_BUILD_LOCK = threading.Lock()


@dataclass
class IndexInfo:
    key: str
    filename: str
    pages: int
    chunks: int
    cached: bool


# ----------------------------------------------------------------- loading
@traceable(name="load_document", run_type="retriever")
def load_document(path: Path, display_name: str | None = None) -> list[Document]:
    name = display_name or path.name
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        docs = [
            Document(page_content=page.extract_text() or "", metadata={"page": i, "total_pages": len(reader.pages)})
            for i, page in enumerate(reader.pages)
        ]
    else:
        docs = [Document(page_content=path.read_text(encoding="utf-8", errors="replace"), metadata={})]
    for d in docs:
        # PDF extraction can produce lone surrogates -> UnicodeEncodeError downstream.
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
        d.metadata["source"] = name
    return [d for d in docs if d.page_content.strip()]


@traceable(name="split_documents")
def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


@traceable(name="build_vectorstore")
def build_vectorstore(chunks: list[Document]) -> FAISS:
    return FAISS.from_documents(chunks, get_embeddings())


# --------------------------------------------------------- cache key logic
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def index_key_for(file_hash: str) -> str:
    meta = {
        "file_sha256": file_hash,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "format": INDEX_FORMAT,
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:32]


def _index_path(key: str) -> Path:
    return settings.index_dir / key


def index_exists(key: str) -> bool:
    return (_index_path(key) / "index.faiss").exists()


def read_meta(key: str) -> dict:
    p = _index_path(key) / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ----------------------------------------------------- build / load (cached)
@traceable(name="load_index", tags=["index"])
def _load_index_run(key: str) -> FAISS:
    # Only indexes this app wrote itself are loaded, so pickle deserialisation is safe.
    return FAISS.load_local(str(_index_path(key)), get_embeddings(), allow_dangerous_deserialization=True)


@lru_cache(maxsize=64)
def load_index(key: str) -> FAISS:
    return _load_index_run(key)


@traceable(name="build_index", tags=["index"])
def _build_index_run(path: Path, key: str, display_name: str) -> IndexInfo:
    docs = load_document(path, display_name)
    if not docs:
        raise ValueError(f"No extractable text found in {display_name}.")
    chunks = split_documents(docs)
    vs = build_vectorstore(chunks)
    out = _index_path(key)
    out.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out))
    meta = {
        "filename": display_name,
        "source_path": str(path),
        "pages": len(docs),
        "chunks": len(chunks),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return IndexInfo(key, display_name, len(docs), len(chunks), cached=False)


@traceable(name="load_or_build_index", tags=["setup"])
def load_or_build_index(path: Path, display_name: str | None = None, force_rebuild: bool = False) -> IndexInfo:
    display_name = display_name or path.name
    key = index_key_for(file_sha256(path))
    with _BUILD_LOCK:
        if index_exists(key) and not force_rebuild:
            meta = read_meta(key)
            return IndexInfo(key, display_name, meta.get("pages", 0), meta.get("chunks", 0), cached=True)
        if force_rebuild:
            load_index.cache_clear()
        return _build_index_run(path, key, display_name)


# ------------------------------------------------------------- per-thread
def ingest_upload(file_bytes: bytes, filename: str, thread_id: str) -> IndexInfo:
    """Persist an uploaded file, index it (or reuse the cache) and attach it to a chat."""
    if not file_bytes:
        raise ValueError("Empty upload.")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {suffix!r}. Use PDF, TXT or MD.")
    digest = hashlib.sha256(file_bytes).hexdigest()
    dest = settings.data_dir / "uploads" / f"{digest}{suffix}"
    if not dest.exists():
        dest.write_bytes(file_bytes)
    info = load_or_build_index(dest, display_name=filename)
    threads.attach_document(thread_id, info.key, filename, info.pages, info.chunks)
    return info


# ---------------------------------------------------------- knowledge base
def knowledge_files() -> list[Path]:
    return sorted(
        p for p in settings.knowledge_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def knowledge_status() -> list[dict]:
    """Which knowledge-base files are indexed (without hashing huge files every rerun)."""
    out = []
    for p in knowledge_files():
        key = _kb_key_cached(str(p), p.stat().st_size, int(p.stat().st_mtime))
        meta = read_meta(key) if index_exists(key) else {}
        out.append(
            {
                "file": p.name,
                "path": str(p),
                "size_mb": round(p.stat().st_size / 1e6, 2),
                "indexed": bool(meta),
                "pages": meta.get("pages"),
                "chunks": meta.get("chunks"),
                "key": key,
            }
        )
    return out


@lru_cache(maxsize=256)
def _kb_key_cached(path: str, size: int, mtime: int) -> str:
    return index_key_for(file_sha256(Path(path)))


def index_knowledge_base(force_rebuild: bool = False, progress=None) -> list[IndexInfo]:
    files = knowledge_files()
    results = []
    for i, p in enumerate(files, 1):
        if progress:
            progress(i, len(files), p.name)
        results.append(load_or_build_index(p, force_rebuild=force_rebuild))
    return results


def add_to_knowledge_base(file_bytes: bytes, filename: str) -> Path:
    dest = settings.knowledge_dir / Path(filename).name
    dest.write_bytes(file_bytes)
    return dest


def remove_from_knowledge_base(filename: str) -> None:
    p = settings.knowledge_dir / Path(filename).name
    if p.exists():
        p.unlink()


def clear_index_cache() -> None:
    load_index.cache_clear()
    shutil.rmtree(settings.index_dir, ignore_errors=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- retrieval
def active_index_keys(thread_id: str | None, include_knowledge: bool = True) -> list[str]:
    keys: list[str] = []
    if thread_id:
        keys += [d["index_key"] for d in threads.thread_documents(thread_id)]
    if include_knowledge:
        keys += [s["key"] for s in knowledge_status() if s["indexed"]]
    seen: set[str] = set()
    return [k for k in keys if index_exists(k) and not (k in seen or seen.add(k))]


@traceable(name="retrieve", run_type="retriever")
def retrieve(query: str, thread_id: str | None = None, k: int | None = None,
             include_knowledge: bool = True) -> list[Document]:
    """Top-k chunks across this chat's uploads and the shared knowledge base."""
    k = k or settings.retrieval_k
    # Indexes are shared by content hash, so label hits with the name used in *this* chat.
    labels = {d["index_key"]: d["filename"] for d in threads.thread_documents(thread_id)} if thread_id else {}
    scored: list[tuple[Document, float]] = []
    for key in active_index_keys(thread_id, include_knowledge):
        try:
            for doc, score in load_index(key).similarity_search_with_score(query, k=k):
                if key in labels:
                    doc = Document(page_content=doc.page_content, metadata={**doc.metadata, "source": labels[key]})
                scored.append((doc, score))
        except Exception as exc:  # corrupt / incompatible index -> skip, don't crash the chat
            log.warning("Skipping index %s: %s", key, exc)
    scored.sort(key=lambda pair: pair[1])  # FAISS L2 distance: lower is closer
    return [d for d, _ in scored[:k]]
