"""Chat page: greeting, conversations, per-chat documents, tool trace, streaming."""
from __future__ import annotations

import html

import streamlit as st

from nexus import memory, service, threads
from nexus.persistence import get_store
from nexus.rag import indexing
from ui.common import chat_options, init_session, pretty, status_footer, user_picker
from ui.theme import name_from_memories, side_title, summarize_tools, time_greeting, tool_running

init_session()
ss = st.session_state
thread_id = ss["thread_id"]

# ================================================================ sidebar
if st.sidebar.button("New chat", key="new_chat", icon=":material/edit_square:", use_container_width=True):
    ss["thread_id"] = threads.new_thread_id()
    st.rerun()

convs = threads.list_threads(ss["user_id"])
side_title("Conversations")
if not convs:
    st.sidebar.caption("Your chats will be listed here.")
for t in convs[:40]:
    c1, c2 = st.sidebar.columns([0.86, 0.14])
    active = t.thread_id == thread_id
    if c1.button(t.title, key=f"open-{t.thread_id}", use_container_width=True, disabled=active):
        ss["thread_id"] = t.thread_id
        st.rerun()
    if c2.button(":material/delete:", key=f"del-{t.thread_id}", help=f"Delete “{t.title}”"):
        service.delete_conversation(t.thread_id)
        if active:
            ss["thread_id"] = threads.new_thread_id()
        st.rerun()

side_title("Documents in this chat")
docs = threads.thread_documents(thread_id)
for d in docs:
    c1, c2 = st.sidebar.columns([0.86, 0.14])
    c1.html(f'<div class="nx-doc">{html.escape(d["filename"])}'
            f'<span>{d["pages"]} pages, {d["chunks"]} passages</span></div>')
    if c2.button(":material/close:", key=f"detach-{d['index_key']}", help="Remove from this chat"):
        threads.detach_document(thread_id, d["index_key"])
        st.rerun()

uploads = st.sidebar.file_uploader(
    "Add a PDF, TXT or Markdown file", type=["pdf", "txt", "md"],
    key=f"upload-{thread_id}", accept_multiple_files=True,
    label_visibility="collapsed" if docs else "visible",
)
for up in uploads or []:
    if up.name in {d["filename"] for d in docs}:
        continue
    threads.upsert_thread(thread_id, ss["user_id"])
    with st.sidebar.status(f"Reading {up.name}…") as box:
        try:
            info = indexing.ingest_upload(up.getvalue(), up.name, thread_id)
            box.update(label=f"{up.name} is ready" + (" (from cache)" if info.cached else ""), state="complete")
        except Exception as exc:
            box.update(label=f"Couldn't read {up.name}: {exc}", state="error")
    st.rerun()

with st.sidebar.expander("Preferences", icon=":material/tune:"):
    ss["use_memory"] = st.toggle("Remember things about me", ss["use_memory"],
                                 help="Learns facts you share and uses them in every chat.")
    ss["include_knowledge"] = st.toggle("Search the knowledge base", ss["include_knowledge"],
                                        help="Document answers also draw on the shared library.")
    ss["follow_ups"] = st.toggle("Suggest follow-up questions", ss["follow_ups"])
    ss["persona"] = st.text_area("How should Nexus respond?", ss["persona"],
                                 placeholder="For example: keep answers short and use Python examples.")

user_picker()
status_footer()

# ================================================================== main
history, summary = service.load_history(thread_id)
info = threads.get_thread(thread_id)


def render_trace(tools: list[dict]) -> None:
    if not tools:
        return
    with st.expander(summarize_tools([t["name"] for t in tools]), icon=":material/manufacturing:"):
        for t in tools:
            st.markdown(f"**{t['name']}**")
            st.code(pretty(t["args"]), language="json")
            if t.get("result"):
                st.code(pretty(t["result"]), language="json")


def starter(label: str, prompt: str, key: str, col, icon: str) -> None:
    if col.button(label, key=f"starter-{key}", icon=icon, use_container_width=True):
        ss["queued_prompt"] = prompt
        st.rerun()


if not history:
    mems = memory.recall(get_store(), ss["user_id"]) if ss["use_memory"] else []
    name = name_from_memories(mems)
    greeting = f"{time_greeting()}, {name}." if name else f"{time_greeting()}."
    st.html(
        f'<div class="nx-hero"><h1>{html.escape(greeting)}</h1>'
        "<p>Ask anything. Nexus searches the web, reads your documents, does exact maths, "
        "and remembers what you tell it.</p></div>"
    )
    c1, c2 = st.columns(2)
    starter("What happened in AI this week?", "What are the most important AI news stories from this week?", "news", c1, ":material/public:")
    starter("Grow ₹50,000 at 7% for 10 years",
            "What does ₹50,000 grow to at 7% interest compounded yearly for 10 years?", "calc", c2, ":material/calculate:")
    if docs:
        starter(f"Summarize {docs[0]['filename']}",
                f"Give me a structured summary of {docs[0]['filename']}.", "doc", c1, ":material/description:")
    else:
        starter("Explain RAG in plain words", "Explain retrieval-augmented generation in plain words.", "rag", c1, ":material/lightbulb:")
    starter("What do you remember about me?", "What do you remember about me?", "mem", c2, ":material/psychology:")
else:
    n_msgs = len(history)
    meta = f"{n_msgs} message{'s' if n_msgs != 1 else ''}"
    if docs:
        meta += f", {len(docs)} document{'s' if len(docs) != 1 else ''} attached"
    st.html(f'<div class="nx-thread-head"><h2>{html.escape(info.title if info else "New chat")}</h2>'
            f"<p>{meta}</p></div>")
    if summary:
        with st.expander("Earlier messages are summarized to keep answers fast", icon=":material/compress:"):
            st.markdown(summary)

for msg in history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_trace(msg.get("tools", []))
        st.markdown(msg["content"])

prompt = st.chat_input("Message Nexus") or ss.pop("queued_prompt", None)
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        trace_area = st.container()
        answer_box = st.empty()
        status_box = None
        text, used, had_error = "", [], False
        for ev in service.stream_turn(thread_id, ss["user_id"], prompt, **chat_options()):
            if ev.kind == "token":
                text += ev.text
                answer_box.markdown(text + " ▍")
            elif ev.kind == "tool_start":
                used.append(ev.name)
                text = ""
                answer_box.empty()
                if status_box is None:
                    status_box = trace_area.status(tool_running(ev.name), expanded=False)
                else:
                    status_box.update(label=tool_running(ev.name), state="running")
                status_box.markdown(f"**{ev.name}**")
                status_box.code(pretty(ev.data, 400), language="json")
            elif ev.kind == "tool_end" and status_box is not None:
                status_box.code(pretty(ev.text, 600), language="json")
            elif ev.kind == "summarized":
                st.toast("Older messages were summarized to keep answers fast.")
            elif ev.kind == "error":
                had_error = True
                st.error(ev.text, icon=":material/error:")
            elif ev.kind == "done":
                text = ev.text or text
        if text or not had_error:
            answer_box.markdown(text or "_No response._")
        if status_box is not None:
            status_box.update(label=summarize_tools(used), state="complete", expanded=False)
    if not had_error:
        st.rerun()  # refresh title, message count and sidebar order
