"""Stand-alone workflows: essay evaluation and report writing."""
from __future__ import annotations

import streamlit as st

from nexus.workflows import evaluate_essay, report_and_summary
from ui.common import status_footer
from ui.theme import page_header

status_footer()
page_header("Workflows", "Focused tools that run a fixed sequence of steps instead of a conversation.")

tab_essay, tab_report = st.tabs([":material/rate_review: Essay feedback", ":material/summarize: Report and summary"])

with tab_essay:
    st.caption("Three reviewers grade language, analysis and clarity at the same time, "
               "then a final pass combines their feedback.")
    with st.form("essay", border=False):
        essay = st.text_area("Essay", height=260, placeholder="Paste your essay here", label_visibility="collapsed")
        go = st.form_submit_button("Get feedback", type="primary", icon=":material/play_arrow:")
    if go and essay.strip():
        with st.spinner("Reviewing your essay…"):
            r = evaluate_essay(essay)
        scores = r.get("individual_scores", [])
        cols = st.columns(4)
        cols[0].metric("Overall", f"{r.get('avg_score', 0):.1f}/10")
        labels = [("Language", "language_feedback"), ("Analysis", "analysis_feedback"), ("Clarity", "clarity_feedback")]
        for col, (label, _), score in zip(cols[1:], labels, scores + [None] * 3):
            col.metric(label, f"{score}/10" if score is not None else "–")
        st.subheader("Summary")
        st.markdown(r.get("overall_feedback", ""))
        for label, key in labels:
            with st.expander(f"{label} feedback"):
                st.write(r.get(key, ""))
    elif go:
        st.warning("Paste an essay first.", icon=":material/edit_note:")

with tab_report:
    st.caption("Writes a detailed report on a topic, then condenses it into key points.")
    with st.form("report", border=False):
        c1, c2 = st.columns([0.72, 0.28], vertical_alignment="bottom")
        topic = c1.text_input("Topic", placeholder="Unemployment in India")
        points = c2.number_input("Key points", 3, 10, 5)
        go = st.form_submit_button("Write report", type="primary", icon=":material/play_arrow:")
    if go and topic.strip():
        with st.spinner("Writing the report…"):
            r = report_and_summary(topic, int(points))
        st.subheader("Key points")
        st.markdown(r["summary"])
        with st.expander("Full report", icon=":material/article:"):
            st.markdown(r["report"])
    elif go:
        st.warning("Enter a topic first.", icon=":material/edit_note:")
