"""Session helpers and sidebar pieces shared by all pages."""
from __future__ import annotations

import html
import json

import streamlit as st

from nexus import threads


def init_session() -> None:
    ss = st.session_state
    ss.setdefault("user_id", "default")
    ss.setdefault("thread_id", None)
    ss.setdefault("use_memory", True)
    ss.setdefault("include_knowledge", True)
    ss.setdefault("follow_ups", False)
    ss.setdefault("persona", "")
    if ss["thread_id"] is None:
        existing = threads.list_threads(ss["user_id"])
        ss["thread_id"] = existing[0].thread_id if existing else threads.new_thread_id()


def chat_options() -> dict:
    ss = st.session_state
    return {
        "use_memory": ss["use_memory"],
        "include_knowledge": ss["include_knowledge"],
        "follow_ups": ss["follow_ups"],
        "persona": ss["persona"].strip(),
    }


def user_picker() -> None:
    """Profile switcher: long-term memory and conversation lists are per profile."""
    ss = st.session_state
    new_user = st.sidebar.text_input(
        "Profile", value=ss["user_id"], key="profile_input",
        help="Each profile keeps its own conversations and memories.",
    )
    new_user = (new_user or "default").strip()
    if new_user != ss["user_id"]:
        ss["user_id"] = new_user
        existing = threads.list_threads(new_user)
        ss["thread_id"] = existing[0].thread_id if existing else threads.new_thread_id()
        st.rerun()


def status_footer() -> None:
    from nexus import service

    s = service.status()
    model = s["chat_model"].split(":", 1)[-1]
    tracing = f"Tracing to LangSmith ({html.escape(s['langsmith_project'])})" if s["tracing"] else "Tracing off"
    st.sidebar.html(
        f'<div class="nx-status"><i></i>{html.escape(model)}<br>'
        f'{s["persistence"]} storage, {s["web_search"]} search<br>{tracing}</div>'
    )
    for err in s["mcp_errors"]:
        st.sidebar.warning(f"MCP server unavailable: {err}")


def pretty(value, limit: int = 1500) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    else:
        text = str(value)
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except Exception:
            pass
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"
