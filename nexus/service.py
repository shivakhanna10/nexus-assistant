"""Application service: one place the Streamlit UI and the CLI both talk to.

Handles graph construction (once), streaming a turn as typed events, loading
history, auto-titling conversations, and LangSmith run metadata.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from . import threads
from .agent import build_agent
from .config import settings
from .llm import get_utility_model
from .mcp_tools import load_mcp_tools
from .persistence import get_checkpointer, get_store
from .tools import builtin_tools

log = logging.getLogger(__name__)


@dataclass
class App:
    graph: object
    tools: list
    mcp_errors: list[str] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


_APP: App | None = None
_APP_LOCK = threading.Lock()


def get_app(reload: bool = False) -> App:
    global _APP
    with _APP_LOCK:
        if _APP is None or reload:
            mcp, errors = load_mcp_tools()
            tools = builtin_tools() + mcp
            graph = build_agent(tools, checkpointer=get_checkpointer(), store=get_store())
            _APP = App(graph=graph, tools=tools, mcp_errors=errors)
        return _APP


# ---------------------------------------------------------------- events
@dataclass
class Event:
    kind: str            # token | tool_start | tool_end | summarized | error | done
    text: str = ""
    name: str = ""
    data: dict | None = None


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # e.g. Anthropic content blocks
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def run_config(thread_id: str, user_id: str, **options) -> dict:
    return {
        "configurable": {"thread_id": thread_id, "user_id": user_id, **options},
        "metadata": {"thread_id": thread_id, "user_id": user_id},
        "tags": ["chat"],
        "run_name": "chat_turn",
        "recursion_limit": 40,
    }


def stream_turn(thread_id: str, user_id: str, text: str, **options) -> Iterator[Event]:
    """Run one user turn, yielding UI-friendly events as they happen."""
    app = get_app()
    threads.upsert_thread(thread_id, user_id)
    config = run_config(thread_id, user_id, **options)
    final_parts: list[str] = []
    try:
        for mode, payload in app.graph.stream(
            {"messages": [HumanMessage(content=text)]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, meta = payload
                # AIMessageChunk when the provider streams; a full AIMessage when it doesn't.
                if meta.get("langgraph_node") == "agent" and isinstance(chunk, (AIMessageChunk, AIMessage)):
                    tok = _text_of(chunk.content)
                    if tok:
                        final_parts.append(tok)
                        yield Event("token", text=tok)
            elif mode == "updates":
                for node, update in (payload or {}).items():
                    if not isinstance(update, dict):
                        continue
                    msgs = update.get("messages", []) or []
                    if node == "agent":
                        for m in msgs:
                            if isinstance(m, AIMessage):
                                for tc in m.tool_calls:
                                    # Separate pre-tool chatter from the final answer.
                                    final_parts.clear()
                                    yield Event("tool_start", name=tc["name"], data=tc.get("args", {}))
                    elif node == "tools":
                        for m in msgs:
                            if isinstance(m, ToolMessage):
                                yield Event("tool_end", name=m.name or "tool", text=_text_of(m.content),
                                            data={"status": getattr(m, "status", "success")})
                    elif node == "summarize":
                        yield Event("summarized", text=update.get("summary", ""))
    except Exception as exc:
        log.exception("Turn failed")
        yield Event("error", text=f"{type(exc).__name__}: {exc}")
        return

    _maybe_autotitle(thread_id, text)
    yield Event("done", text="".join(final_parts))


def _maybe_autotitle(thread_id: str, first_message: str) -> None:
    info = threads.get_thread(thread_id)
    if not info or info.title != "New chat":
        return
    title = first_message.strip().splitlines()[0][:48] if first_message.strip() else "New chat"
    try:
        out = get_utility_model().invoke(
            "Write a 3-6 word title for a chat that starts with this message. "
            f"Reply with the title only, no quotes.\n\nMessage: {first_message[:1000]}"
        )
        title = _text_of(out.content).strip().strip('"\'')[:60] or title
    except Exception:
        pass
    threads.rename_thread(thread_id, title)


# --------------------------------------------------------------- history
def load_history(thread_id: str) -> tuple[list[dict], str]:
    """Return (display messages, rolling summary) for a thread."""
    state = get_app().graph.get_state({"configurable": {"thread_id": thread_id}})
    values = state.values or {}
    out: list[dict] = []
    pending_tools: list[dict] = []
    for m in values.get("messages", []):
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _text_of(m.content), "tools": []})
        elif isinstance(m, AIMessage):
            for tc in m.tool_calls:
                pending_tools.append({"name": tc["name"], "args": tc.get("args", {}), "result": ""})
            text = _text_of(m.content)
            if text and not m.tool_calls:
                out.append({"role": "assistant", "content": text, "tools": pending_tools})
                pending_tools = []
        elif isinstance(m, ToolMessage):
            for t in pending_tools:
                if t["name"] == m.name and not t["result"]:
                    t["result"] = _text_of(m.content)
                    break
    return out, values.get("summary", "")


def delete_conversation(thread_id: str) -> None:
    try:
        get_checkpointer().delete_thread(thread_id)
    except Exception as exc:
        log.warning("Could not delete checkpoints for %s: %s", thread_id, exc)
    threads.delete_thread(thread_id)


def status() -> dict:
    from . import web
    from .persistence import backend_name

    app = get_app()
    return {
        "chat_model": settings.chat_model,
        "utility_model": settings.utility_model,
        "embedding_model": settings.embedding_model,
        "persistence": backend_name(),
        "web_search": web.provider_name(),
        "tracing": settings.tracing_enabled,
        "langsmith_project": settings.langsmith_project,
        "tools": app.tool_names,
        "mcp_errors": app.mcp_errors,
        "semantic_memory": settings.semantic_memory,
    }
