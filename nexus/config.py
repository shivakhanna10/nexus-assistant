"""Central configuration, read once from environment variables / .env.

Every tunable that was hard-coded across the original projects (model names,
chunk sizes, CRAG thresholds, memory limits, DB locations, API keys) lives here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    p = Path(raw) if raw else default
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p


@dataclass(frozen=True)
class Settings:
    # ---- Models (any provider supported by langchain's init_chat_model) ----
    chat_model: str = os.getenv("CHAT_MODEL", "openai:gpt-4o-mini")
    utility_model: str = os.getenv("UTILITY_MODEL", "openai:gpt-4o-mini")  # graders, memory, rewrite
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    temperature: float = _float("CHAT_TEMPERATURE", 0.3)

    # ---- Storage ----
    data_dir: Path = _path("DATA_DIR", ROOT_DIR / "data")
    knowledge_dir: Path = _path("KNOWLEDGE_DIR", ROOT_DIR / "data" / "knowledge")
    postgres_uri: str | None = os.getenv("POSTGRES_URI") or None

    # ---- RAG / indexing ----
    chunk_size: int = _int("CHUNK_SIZE", 1000)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)
    retrieval_k: int = _int("RETRIEVAL_K", 4)

    # ---- Corrective RAG ----
    crag_upper_threshold: float = _float("CRAG_UPPER_THRESHOLD", 0.7)
    crag_lower_threshold: float = _float("CRAG_LOWER_THRESHOLD", 0.3)
    crag_refine: bool = _bool("CRAG_REFINE", True)        # sentence-level strip filtering
    crag_web_fallback: bool = _bool("CRAG_WEB_FALLBACK", True)
    max_concurrency: int = _int("MAX_CONCURRENCY", 8)

    # ---- Short-term memory ----
    max_context_tokens: int = _int("MAX_CONTEXT_TOKENS", 6000)  # trimming budget per LLM call
    summarize_after: int = _int("SUMMARIZE_AFTER_MESSAGES", 24)  # summarise when history grows past this
    keep_recent: int = _int("KEEP_RECENT_MESSAGES", 6)           # messages kept verbatim after summarising

    # ---- Long-term memory ----
    long_term_memory: bool = _bool("LONG_TERM_MEMORY", True)
    semantic_memory: bool = _bool("SEMANTIC_MEMORY", False)  # embed memories for similarity search
    memory_recall_limit: int = _int("MEMORY_RECALL_LIMIT", 20)

    # ---- Tool API keys (never hard-code these) ----
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY") or None
    alphavantage_api_key: str | None = os.getenv("ALPHAVANTAGE_API_KEY") or None
    weatherstack_api_key: str | None = os.getenv("WEATHERSTACK_API_KEY") or None
    mcp_config_path: Path = _path("MCP_CONFIG", ROOT_DIR / "mcp_servers.json")

    # ---- Observability ----
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "nexus-chatbot"

    extra: dict = field(default_factory=dict)

    @property
    def tracing_enabled(self) -> bool:
        return _bool("LANGSMITH_TRACING", False) or _bool("LANGCHAIN_TRACING_V2", False)

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indices"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "nexus.db"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.knowledge_dir, self.index_dir, self.data_dir / "uploads"):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
