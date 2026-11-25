"""Model factories.

The original projects instantiated ``ChatOpenAI()`` in every file. Here models are
created in one place via ``init_chat_model`` so you can switch provider with an
env var (e.g. ``CHAT_MODEL=anthropic:claude-sonnet-4-5`` after installing
``langchain-anthropic``) without touching code.
"""
from __future__ import annotations

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import settings


@lru_cache(maxsize=8)
def get_chat_model(model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    """Main conversational model (streams to the UI)."""
    return init_chat_model(
        model or settings.chat_model,
        temperature=settings.temperature if temperature is None else temperature,
    )


@lru_cache(maxsize=1)
def get_utility_model() -> BaseChatModel:
    """Deterministic model for graders, query rewriting and memory extraction."""
    return init_chat_model(settings.utility_model, temperature=0)


@lru_cache(maxsize=2)
def get_embeddings(model: str | None = None) -> Embeddings:
    name = model or settings.embedding_model
    if ":" in name:
        from langchain.embeddings import init_embeddings

        return init_embeddings(name)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=name)


def embedding_dims(model: str | None = None) -> int:
    """Vector size, needed by the semantic long-term memory index."""
    name = (model or settings.embedding_model).split(":", 1)[-1].removeprefix("models/")
    known = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "gemini-embedding-001": 3072,
        "text-embedding-004": 768,
    }
    if name in known:
        return known[name]
    # Unknown model: measure it once instead of guessing.
    return len(get_embeddings(model).embed_query("dimension probe"))
