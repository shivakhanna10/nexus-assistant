"""Retrieval inspector: run Corrective RAG and see every decision it makes."""
from __future__ import annotations

import html

import streamlit as st

from nexus import threads
from nexus.config import settings
from nexus.rag.crag import run_crag
from ui.common import init_session, status_footer
from ui.theme import page_header, passage

init_session()
status_footer()
page_header(
    "Retrieval inspector",
    "See how Nexus answers from documents: it grades every passage it finds, searches the web when "
    "they fall short, keeps only the sentences that help, then writes the answer.",
)

has_docs = bool(threads.thread_documents(st.session_state["thread_id"]))
c1, c2 = st.columns(2)
scope_thread = c1.toggle("Documents in the current chat", value=has_docs)
include_kb = c2.toggle("Knowledge base", value=True)

with st.form("crag", border=False):
    q = st.text_input("Question", placeholder="Batch normalization vs layer normalization")
    run = st.form_submit_button("Run retrieval", type="primary", icon=":material/play_arrow:")

VERDICTS = {
    "CORRECT": ("correct", ":material/check_circle:", "The documents answer this"),
    "INCORRECT": ("incorrect", ":material/travel_explore:", "The documents don't answer this, so the web was searched"),
    "AMBIGUOUS": ("ambiguous", ":material/help:", "The documents partly answer this, so the web was added"),
}

if run and q:
    with st.status("Retrieving and grading passages…", expanded=False) as box:
        res = run_crag(q, thread_id=st.session_state["thread_id"] if scope_thread else None,
                       include_knowledge=include_kb)
        box.update(label="Retrieval finished", state="complete")

    cls, _, text = VERDICTS.get(res.get("verdict"), ("ambiguous", "", res.get("verdict", "")))
    st.html(f'<span class="nx-verdict {cls}">{html.escape(text)}</span>')
    st.caption(f"{res.get('reason', '')} Thresholds: pass above {settings.crag_upper_threshold}, "
               f"fail below {settings.crag_lower_threshold}.")

    st.subheader("Answer")
    st.markdown(res.get("answer", ""))
    if res.get("sources"):
        st.caption("Sources: " + "; ".join(
            f"[{s['label']}]({s['url']})" if s.get("url") else f"{s['label']} {s['ref']}".strip()
            for s in res["sources"]
        ))

    st.subheader("How it got there")
    docs, scores = res.get("docs", []), res.get("scores", [])
    with st.expander(f"Graded {len(docs)} passages", icon=":material/grading:", expanded=bool(docs)):
        if not docs:
            st.write("No indexed passages matched. Attach a document or index the knowledge base.")
        for d, s in zip(docs, scores):
            page = d.metadata.get("page")
            where = html.escape(str(d.metadata.get("source"))) + (f", page {page + 1}" if isinstance(page, int) else "")
            st.html(f'<div class="nx-score"><div class="bar"><span style="width:{s * 100:.0f}%"></span></div>'
                    f"<b>{s:.2f}</b> {where}</div>")
            passage(d.page_content, 500)
    if res.get("web_query"):
        with st.expander(f"Searched the web for “{res['web_query']}”", icon=":material/travel_explore:"):
            for d in res.get("web_docs", []):
                st.markdown(f"**[{d.metadata.get('title') or d.metadata.get('url')}]({d.metadata.get('url')})**")
                passage(d.page_content.split("CONTENT:", 1)[-1].strip(), 400)
    if res.get("strips"):
        kept = set(res.get("kept_strips", []))
        with st.expander(f"Kept {len(kept)} of {len(res['strips'])} sentences", icon=":material/filter_alt:"):
            for s in res["strips"]:
                st.markdown(f":material/check: {s}" if s in kept else f":gray[:material/close: ~~{s}~~]")
