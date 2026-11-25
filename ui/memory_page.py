"""Long-term memory: what Nexus has learned about the current profile."""
from __future__ import annotations

import datetime as dt
import html

import streamlit as st

from nexus import memory
from nexus.config import settings
from nexus.persistence import get_store
from ui.common import init_session, status_footer, user_picker
from ui.theme import page_header

init_session()
user_picker()
status_footer()
user_id = st.session_state["user_id"]
store = get_store()

page_header(
    "Memory",
    f"What Nexus has learned about the “{user_id}” profile. It picks up facts you share in any chat, "
    "skips duplicates, and uses them to personalize answers.",
)

with st.form("add-memory", clear_on_submit=True, border=False):
    c1, c2 = st.columns([0.8, 0.2], vertical_alignment="bottom")
    new = c1.text_input("Teach Nexus something", placeholder="I prefer answers with bullet points")
    if c2.form_submit_button("Remember", icon=":material/add:", use_container_width=True) and new.strip():
        memory.add_memory(store, user_id, new)
        st.rerun()

items = memory.list_memories(store, user_id)
if not items:
    st.info(
        "Nothing remembered yet. Tell Nexus about yourself in a chat, for example "
        "“I'm a data scientist learning LangGraph and I like concise answers.”",
        icon=":material/psychology:",
    )

for it in items:
    c1, c2 = st.columns([0.92, 0.08], vertical_alignment="center")
    when = it["created_at"].strftime("%d %b %Y") if isinstance(it["created_at"], dt.datetime) else ""
    c1.html(f'<div class="nx-memory"><b>{html.escape(it["text"])}</b><span>Learned {when}</span></div>')
    if c2.button(":material/delete:", key=f"mem-{it['id']}", help="Forget this"):
        memory.delete_memory(store, user_id, it["id"])
        st.rerun()

if items:
    st.write("")
    with st.popover("Forget everything", icon=":material/delete_sweep:"):
        st.write(f"This permanently deletes all {len(items)} memories for “{user_id}”.")
        if st.button("Delete all memories", type="primary"):
            memory.clear_memories(store, user_id)
            st.rerun()

    if settings.semantic_memory:
        st.subheader("Search memories")
        q = st.text_input("Search memories", placeholder="What is the user learning?", label_visibility="collapsed")
        if q:
            for m in memory.recall(store, user_id, query=q)[:5]:
                st.write("•", m)
