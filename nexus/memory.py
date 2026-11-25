"""Short-term + long-term memory (from memory-in-llms, notebooks 1-8).

Short-term (per thread, in the checkpointer):
  * token-budget trimming before every model call            (3_stm_trimming)
  * rolling summary + deletion of old messages               (4_stm_deletion, 5_stm_summarization)
    – the cut point is moved so a tool call is never separated from its result.

Long-term (per user, in the store, shared by all threads):
  * LLM memory extraction with duplicate detection           (7_ltm_implementation)
  * optional semantic recall over embedded memories          (6_ltm_basics)
  * persistent store in SQLite or Postgres                   (8_ltm_postgres)
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from langchain_core.messages import (
    AnyMessage, HumanMessage, RemoveMessage, SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from .config import settings
from .llm import get_utility_model, message_text

# ======================================================== long-term memory
MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence written in the third person ("User ...").
- No speculation; only facts stated by the user. Ignore questions and one-off requests.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""


class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")


class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: list[MemoryItem] = Field(default_factory=list, description="Atomic user memories")


@lru_cache(maxsize=1)
def _extractor():
    return get_utility_model().with_structured_output(MemoryDecision)


def namespace(user_id: str) -> tuple[str, ...]:
    return ("user", user_id, "details")


def list_memories(store: BaseStore, user_id: str) -> list[dict]:
    items = store.search(namespace(user_id), limit=500)
    items = sorted(items, key=lambda it: it.created_at)
    return [{"id": it.key, "text": it.value.get("data", ""), "created_at": it.created_at} for it in items]


def recall(store: BaseStore, user_id: str, query: str | None = None) -> list[str]:
    ns = namespace(user_id)
    if settings.semantic_memory and query:
        items = store.search(ns, query=query, limit=settings.memory_recall_limit)
    else:
        items = store.search(ns, limit=500)
    return [it.value["data"] for it in items if it.value.get("data")]


def add_memory(store: BaseStore, user_id: str, text: str) -> str:
    key = str(uuid.uuid4())
    store.put(namespace(user_id), key, {"data": text.strip()})
    return key


def delete_memory(store: BaseStore, user_id: str, key: str) -> None:
    store.delete(namespace(user_id), key)


def clear_memories(store: BaseStore, user_id: str) -> int:
    items = list_memories(store, user_id)
    for it in items:
        delete_memory(store, user_id, it["id"])
    return len(items)


def extract_and_store(store: BaseStore, user_id: str, user_text: str) -> list[str]:
    """Run the memory extractor on one user message; store and return only new facts."""
    if not user_text.strip():
        return []
    existing = list_memories(store, user_id)
    existing_txt = "\n".join(f"- {m['text']}" for m in existing) or "(empty)"
    decision: MemoryDecision = _extractor().invoke([
        SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing_txt)),
        HumanMessage(content=f"USER MESSAGE:\n{user_text}"),
    ])
    saved: list[str] = []
    known = {m["text"].strip().lower() for m in existing}
    if decision.should_write:
        for mem in decision.memories:
            text = mem.text.strip()
            if mem.is_new and text and text.lower() not in known:
                add_memory(store, user_id, text)
                known.add(text.lower())
                saved.append(text)
    return saved


# ======================================================= short-term memory
def trim_for_model(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Keep the most recent messages that fit the token budget, starting on a human turn."""
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=settings.max_context_tokens,
        start_on="human",
        allow_partial=False,
    )
    return trimmed or messages[-1:]


def safe_cut_index(messages: list[AnyMessage], keep_recent: int) -> int:
    """Index where the verbatim tail starts; always lands on a HumanMessage so
    AI tool-calls and their ToolMessages are never split apart."""
    start = max(len(messages) - keep_recent, 0)
    # Prefer moving forward (keeps at most `keep_recent` messages) ...
    for i in range(start, len(messages)):
        if isinstance(messages[i], HumanMessage):
            return i
    # ... otherwise move back to the turn that started the current exchange.
    for i in range(start - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


def summarize_history(messages: list[AnyMessage], existing_summary: str, model) -> dict:
    """Fold old messages into a running summary and delete them from state."""
    cut = safe_cut_index(messages, settings.keep_recent)
    if cut <= 0:
        return {}
    old = messages[:cut]
    instruction = (
        f"Existing summary:\n{existing_summary}\n\nExtend the summary using the new conversation above."
        if existing_summary
        else "Summarize the conversation above."
    ) + " Be concise but keep names, facts, decisions, numbers and open questions."
    # Flatten to plain text: replaying raw AI tool-call messages without their
    # ToolMessages would be rejected by the provider.
    lines = []
    for m in old:
        text = message_text(m)
        if m.type == "human":
            lines.append(f"User: {text}")
        elif m.type == "ai" and text:
            lines.append(f"Assistant: {text}")
        elif m.type == "tool":
            lines.append(f"Tool[{getattr(m, 'name', 'tool')}]: {text[:500]}")
    response = model.invoke([
        SystemMessage(content="You summarise conversations for an assistant's memory."),
        HumanMessage(content="Conversation:\n" + "\n".join(lines) + "\n\n" + instruction),
    ])
    return {
        "summary": message_text(response),
        "messages": [RemoveMessage(id=m.id) for m in old if m.id],
    }
