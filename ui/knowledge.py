"""Knowledge base: shared documents every chat can search."""
from __future__ import annotations

import streamlit as st

from nexus.rag import indexing
from ui.common import status_footer
from ui.theme import page_header, passage

status_footer()
page_header(
    "Knowledge base",
    "A shared library every chat can search. Each file is indexed once, then loads instantly from cache.",
)

with st.form("kb-add", clear_on_submit=True, border=False):
    ups = st.file_uploader("Add files to the library", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.form_submit_button("Add to library", icon=":material/add:") and ups:
        for up in ups:
            indexing.add_to_knowledge_base(up.getvalue(), up.name)
        st.toast(f"Added {len(ups)} file{'s' if len(ups) != 1 else ''}. Index them to make them searchable.")
        st.rerun()

with st.spinner("Checking the library…"):
    rows = indexing.knowledge_status()

if not rows:
    st.info(
        "The library is empty. Add files above, or copy PDFs into the `data/knowledge` folder: "
        "for example the books in `corrective-rag-main/documents`.",
        icon=":material/library_add:",
    )
else:
    pending = [r for r in rows if not r["indexed"]]
    m1, m2, m3 = st.columns(3)
    m1.metric("Files", len(rows))
    m2.metric("Searchable", len(rows) - len(pending))
    m3.metric("Passages", sum(r["chunks"] or 0 for r in rows))

    st.dataframe(
        [
            {"File": r["file"], "Size (MB)": r["size_mb"], "Status": "Searchable" if r["indexed"] else "Not indexed",
             "Pages": r["pages"], "Passages": r["chunks"]}
            for r in rows
        ],
        use_container_width=True, hide_index=True,
    )

    c1, c2, c3 = st.columns([1.2, 1, 1.4])
    if c1.button(f"Index {len(pending)} file{'s' if len(pending) != 1 else ''}", type="primary",
                 icon=":material/bolt:", disabled=not pending, use_container_width=True):
        bar = st.progress(0.0, text="Starting…")
        indexing.index_knowledge_base(
            progress=lambda i, n, name: bar.progress((i - 1) / n, text=f"Indexing {name} ({i} of {n})")
        )
        bar.progress(1.0, text="Done")
        st.rerun()
    if c2.button("Rebuild all", icon=":material/refresh:", use_container_width=True):
        with st.spinner("Rebuilding every index…"):
            indexing.index_knowledge_base(force_rebuild=True)
        st.rerun()
    with c3.popover("Remove a file", icon=":material/delete:", use_container_width=True):
        target = st.selectbox("File", [r["file"] for r in rows], label_visibility="collapsed")
        if st.button(f"Remove {target}", type="primary"):
            indexing.remove_from_knowledge_base(target)
            st.rerun()

st.subheader("Try a search")
q = st.text_input("Search the library", placeholder="What is the bias-variance tradeoff?",
                  label_visibility="collapsed")
if q:
    hits = indexing.retrieve(q, thread_id=None, include_knowledge=True)
    if not hits:
        st.warning("Nothing to search yet. Index at least one file first.", icon=":material/search_off:")
    for d in hits:
        page = d.metadata.get("page")
        where = f"{d.metadata.get('source')}" + (f", page {page + 1}" if isinstance(page, int) else "")
        with st.expander(where, icon=":material/article:"):
            passage(d.page_content)
