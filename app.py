"""Nexus: unified LangGraph assistant.  Run:  streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Nexus", page_icon=":material/explore:", layout="centered")


def _export_streamlit_secrets() -> None:
    """On Streamlit Community Cloud keys live in st.secrets, not .env. Copy top-level
    string values into the environment *before* nexus reads its settings."""
    import os

    try:
        for key, value in st.secrets.to_dict().items():
            if isinstance(value, (str, int, float, bool)) and not os.environ.get(key):
                os.environ[key] = str(value)
    except Exception:
        pass  # no secrets.toml locally: .env is used instead


_export_streamlit_secrets()

from nexus import service  # noqa: E402
from ui.theme import inject_css, page_header, sidebar_brand  # noqa: E402

inject_css()
sidebar_brand()


def _require_password() -> None:
    """Optional gate: set APP_PASSWORD (env or secrets) to keep a public deployment private."""
    import hmac
    import os

    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("authenticated"):
        return
    page_header("Sign in to Nexus", "This workspace is private. Enter the access password to continue.")
    with st.form("login", border=False):
        attempt = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary", icon=":material/lock_open:"):
            if hmac.compare_digest(attempt.encode(), expected.encode()):
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("That password isn't right. Check it and try again.", icon=":material/lock:")
    st.stop()


_require_password()

try:
    service.get_app()
except Exception as exc:  # most often: missing API key
    page_header("Connect a model to start", "Nexus couldn't reach the configured language model.")
    st.error(f"{exc}")
    st.markdown(
        "1. Open `.env` in the project folder (copy it from `.env.example` if it's missing).\n"
        "2. Paste your key for the configured provider: `GOOGLE_API_KEY` for `google_genai:` models "
        "(free at aistudio.google.com) or `OPENAI_API_KEY` for `openai:` models.\n"
        "3. Restart with `streamlit run app.py`."
    )
    st.stop()

PAGES = [
    st.Page("ui/chat.py", title="Chat", icon=":material/forum:", default=True),
    st.Page("ui/knowledge.py", title="Knowledge base", icon=":material/library_books:"),
    st.Page("ui/memory_page.py", title="Memory", icon=":material/psychology:"),
    st.Page("ui/crag_page.py", title="Retrieval inspector", icon=":material/troubleshoot:"),
    st.Page("ui/workflows_page.py", title="Workflows", icon=":material/account_tree:"),
]
current = st.navigation(PAGES, position="hidden")  # rendered below the brand instead
with st.sidebar.container(key="nav"):
    for i, page in enumerate(PAGES):
        state = "active" if page.url_path == current.url_path else "idle"
        with st.container(key=f"navlink-{state}-{i}"):
            st.page_link(page)
current.run()
