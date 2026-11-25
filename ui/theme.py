"""Visual layer on top of .streamlit/config.toml: brand mark, layout and component polish.

Design direction: a quiet "reading room": spruce sidebar, cool paper canvas,
patina green for actions and a single brass highlight (brand mark + where you are).
Boldness is spent in one place: the large, personal greeting on an empty chat.
"""
from __future__ import annotations

import base64
import html
import re
from datetime import datetime

import streamlit as st

SPRUCE, SPRUCE_2, PAPER, INK, PATINA, BRASS, MIST = (
    "#17322C", "#21423A", "#F2F4F1", "#1A2421", "#1F6F5C", "#D4A23F", "#DCE5E0",
)
MUTED = "#5B6B65"

_CSS = f"""
<style>
:root {{
  --spruce:{SPRUCE}; --spruce-2:{SPRUCE_2}; --paper:{PAPER}; --ink:{INK};
  --patina:{PATINA}; --brass:{BRASS}; --mist:{MIST}; --muted:{MUTED};
}}

/* ---------- canvas ---------- */
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{
  max-width: 820px; padding-top: 2.5rem; padding-bottom: 6rem;
}}
[data-testid="stBottomBlockContainer"] {{ max-width: 820px; padding-bottom: 1.25rem; }}
[data-testid="stBottom"] > div {{ background: linear-gradient(to top, var(--paper) 70%, transparent); }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
p, li {{ line-height: 1.62; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ border-right: 1px solid #0E211D; }}
[data-testid="stSidebarHeader"] {{ height: 2.4rem; min-height: 0; padding-bottom: 0; }}
.st-key-nav {{ gap: 2px; margin-bottom: 1.1rem; }}
.st-key-nav [data-testid="stPageLink-NavLink"] {{
  border-radius: 8px; padding: .38rem .6rem; margin: 0; background: transparent;
}}
.st-key-nav [data-testid="stPageLink-NavLink"] p,
.st-key-nav [data-testid="stPageLink-NavLink"] span {{ color: #C9D6D0; font-weight: 500; }}
.st-key-nav [data-testid="stPageLink-NavLink"]:hover {{ background: var(--spruce-2); }}
[class*="st-key-navlink-active"] [data-testid="stPageLink-NavLink"] {{
  background: var(--spruce-2); box-shadow: inset 3px 0 0 var(--brass);
}}
[class*="st-key-navlink-active"] [data-testid="stPageLink-NavLink"] p,
[class*="st-key-navlink-active"] [data-testid="stPageLink-NavLink"] span {{ color: #FFFFFF; font-weight: 600; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: #9FB3AB; font-size: .82rem; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small {{ color: #9FB3AB; }}

.nx-brand {{ display:flex; align-items:center; gap:.6rem; margin:.1rem 0 1.1rem; }}
.nx-brand svg {{ flex: none; }}
.nx-brand b {{
  font-family: "Bricolage Grotesque", sans-serif; font-size: 1.35rem; font-weight: 650;
  letter-spacing: -0.02em; color: #FFFFFF;
}}
.nx-side-title {{
  color:#9FB3AB; font-size:.8rem; font-weight:600; margin: 1.1rem 0 .35rem;
}}

/* "New chat": the brass action */
.st-key-new_chat button {{
  background: var(--brass); border: none; color: var(--spruce); font-weight: 650;
}}
.st-key-new_chat button:hover {{ background: #E2B455; color: var(--spruce); }}
.st-key-new_chat button p {{ color: var(--spruce); }}

/* conversation list: quiet text rows, brass marker on the open one */
[class*="st-key-open-"] button {{
  background: transparent; border: none; justify-content: flex-start;
  padding: .38rem .6rem; min-height: 0; border-radius: 8px;
}}
[class*="st-key-open-"] button p {{
  color: #C9D6D0; font-weight: 450; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  text-align: left;
}}
[class*="st-key-open-"] button > div, [class*="st-key-starter-"] button > div {{
  justify-content: flex-start; width: 100%; min-width: 0;
}}
[class*="st-key-open-"] [data-testid="stMarkdownContainer"],
[class*="st-key-starter-"] [data-testid="stMarkdownContainer"] {{ text-align: left; min-width: 0; }}
[class*="st-key-open-"] button:hover {{ background: var(--spruce-2); }}
[class*="st-key-open-"] button:disabled {{
  background: var(--spruce-2); box-shadow: inset 3px 0 0 var(--brass); opacity: 1; cursor: default;
}}
[class*="st-key-open-"] button:disabled p {{ color: #FFFFFF; font-weight: 600; }}
[class*="st-key-del-"] button, [class*="st-key-detach-"] button {{
  background: transparent; border: none; min-height: 0; padding: .38rem .3rem; opacity: .45;
}}
[class*="st-key-del-"] button:hover, [class*="st-key-detach-"] button:hover {{ opacity: 1; background: transparent; }}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{ gap: .25rem; align-items: center; }}

.nx-doc {{ font-size:.86rem; color:#E4ECE8; line-height:1.35; padding:.15rem 0; }}
.nx-doc span {{ display:block; color:#9FB3AB; font-size:.78rem; }}
.nx-status {{ font-size:.78rem; color:#8AA199; line-height:1.5; margin-top:1.4rem;
  padding-top:.8rem; border-top:1px solid #2E5248; }}
.nx-status i {{ display:inline-block; width:7px; height:7px; border-radius:50%;
  background:#5FBF8F; margin-right:.4rem; font-style:normal; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
  background: transparent; border: 1px dashed #3B6157;
}}

/* ---------- greeting (the one bold moment) ---------- */
.nx-hero {{ margin: 9vh 0 1.6rem; animation: nx-rise .6s cubic-bezier(.2,.7,.2,1) both; }}
.nx-hero h1 {{
  font-family:"Bricolage Grotesque", sans-serif; font-size: clamp(2.4rem, 5vw, 3.6rem);
  font-weight: 650; line-height: 1.02; letter-spacing: -0.035em; color: var(--spruce);
  margin: 0 0 .7rem; padding: 0;
}}
.nx-hero p {{ font-size: 1.08rem; color: var(--muted); max-width: 34rem; margin: 0; }}
@keyframes nx-rise {{ from {{ opacity:0; transform: translateY(8px); }} to {{ opacity:1; transform:none; }} }}
@media (prefers-reduced-motion: reduce) {{ .nx-hero {{ animation: none; }} }}

[class*="st-key-starter-"] button {{
  background: #FFFFFF; border: 1px solid var(--mist); border-radius: 12px;
  justify-content: flex-start; text-align: left; padding: .75rem .9rem; min-height: 3.6rem;
  transition: border-color .15s ease, background .15s ease;
}}
[class*="st-key-starter-"] button p {{ color: var(--ink); font-weight: 500; text-align:left; }}
[class*="st-key-starter-"] button [data-testid="stIconMaterial"] {{ color: var(--patina); }}
[class*="st-key-starter-"] button:hover {{ border-color: var(--patina); background: #FAFCFB; }}

/* ---------- conversation ---------- */
.nx-thread-head h2 {{
  font-family:"Bricolage Grotesque", sans-serif; font-size: 1.55rem; font-weight: 650;
  letter-spacing:-0.02em; color: var(--spruce); margin: 0; padding: 0;
}}
.nx-thread-head p {{ color: var(--muted); font-size: .88rem; margin: .2rem 0 1.4rem; }}

[data-testid="stChatMessage"] {{ background: transparent; padding: .35rem 0; gap: .8rem; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
  flex-direction: row-reverse;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
  background: #E3EBE7; border-radius: 16px 16px 4px 16px; padding: .6rem 1rem;
  max-width: 82%; margin-left: auto; margin-right: 0; flex: 0 1 auto;
}}
/* headings inside answers stay in proportion to the conversation */
[data-testid="stChatMessageContent"] h1 {{ font-size: 1.3rem; padding: .9rem 0 .3rem; }}
[data-testid="stChatMessageContent"] h2 {{ font-size: 1.15rem; padding: .8rem 0 .25rem; }}
[data-testid="stChatMessageContent"] h3,
[data-testid="stChatMessageContent"] h4 {{ font-size: 1.02rem; padding: .7rem 0 .2rem; }}
[data-testid="stChatMessageContent"] hr {{ margin: .9rem 0; border-color: var(--mist); }}
[data-testid="stChatMessageAvatarUser"] {{ display: none; }}
[data-testid="stChatMessageAvatarAssistant"] {{
  background: var(--spruce); color: var(--brass); border-radius: 9px;
}}
[data-testid="stChatMessageAvatarAssistant"] svg {{ fill: var(--brass); color: var(--brass); }}

/* tool trace */
[data-testid="stChatMessage"] [data-testid="stExpander"] details {{
  border: 1px solid var(--mist); border-radius: 10px; background: #F8FAF9;
}}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary p {{
  font-size: .84rem; color: var(--muted); font-weight: 500;
}}

/* chat input */
[data-testid="stChatInput"] {{
  border: 1px solid #CCD8D2; border-radius: 14px; background: #FFFFFF;
  box-shadow: 0 6px 24px -12px rgba(23, 50, 44, .35);
}}
[data-testid="stChatInput"]:focus-within {{ border-color: var(--patina); }}

/* ---------- pages ---------- */
.nx-page-head {{ margin-bottom: 1.6rem; }}
.nx-page-head h1 {{
  font-family:"Bricolage Grotesque", sans-serif; font-size: 2.1rem; font-weight: 650;
  letter-spacing:-0.03em; color: var(--spruce); margin:0 0 .35rem; padding:0;
}}
.nx-page-head p {{ color: var(--muted); margin:0; max-width: 40rem; }}

.nx-verdict {{ display:inline-flex; align-items:center; gap:.5rem; font-weight:650;
  font-size:.95rem; padding:.35rem .75rem; border-radius: 8px; }}
.nx-verdict.correct {{ background:#E1F0EA; color:#185A4A; }}
.nx-verdict.incorrect {{ background:#F6E3E1; color:#8E3229; }}
.nx-verdict.ambiguous {{ background:#F6EBD3; color:#7A5410; }}

.nx-score {{ display:flex; align-items:center; gap:.6rem; font-size:.85rem; color:var(--muted); }}
.nx-score .bar {{ flex:none; width:90px; height:6px; border-radius:3px; background:var(--mist); overflow:hidden; }}
.nx-score .bar > span {{ display:block; height:100%; background:var(--patina); }}

.nx-memory {{ padding:.7rem 0; border-bottom:1px solid var(--mist); }}
.nx-memory b {{ font-weight:500; color:var(--ink); }}
.nx-memory span {{ display:block; font-size:.78rem; color:var(--muted); margin-top:.15rem; }}

[data-testid="stMainBlockContainer"] h3 {{
  font-size: 1.15rem; padding-top: 1.6rem; border-top: 1px solid var(--mist); margin-top: 1rem;
}}
[data-testid="stMetricValue"] {{ font-family:"Bricolage Grotesque", sans-serif; color: var(--spruce); }}

/* ---------- accessibility ---------- */
button:focus-visible, a:focus-visible, textarea:focus-visible, input:focus-visible {{
  outline: 2px solid var(--brass) !important; outline-offset: 2px;
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding-top: 3.4rem; }}  /* clear the sidebar toggle */
  .nx-hero {{ margin-top: 4vh; }}
}}
</style>
"""

_MARK = f"""<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 30 30">
<rect width="30" height="30" rx="8" fill="{SPRUCE_2}"/>
<circle cx="15" cy="15" r="9" fill="none" stroke="#3B6157" stroke-width="1.4"/>
<path d="M15 5.5 L18 15 L15 24.5 L12 15 Z" fill="{BRASS}"/>
<path d="M15 15 L18 15 L15 24.5 Z" fill="#9C7426"/>
<circle cx="15" cy="15" r="1.6" fill="{SPRUCE}"/></svg>"""


def inject_css() -> None:
    st.html(_CSS)


def sidebar_brand() -> None:
    # Inline <svg> is sanitised away by st.html, so ship the mark as a data URI image.
    src = "data:image/svg+xml;base64," + base64.b64encode(_MARK.encode()).decode()
    st.sidebar.html(f'<div class="nx-brand"><img src="{src}" width="30" height="30" alt=""><b>Nexus</b></div>')


def side_title(text: str) -> None:
    st.sidebar.html(f'<div class="nx-side-title">{html.escape(text)}</div>')


def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.html(f'<div class="nx-page-head"><h1>{html.escape(title)}</h1>{sub}</div>')


def time_greeting() -> str:
    h = datetime.now().hour
    return "Good morning" if 5 <= h < 12 else "Good afternoon" if h < 17 else "Good evening"


_NAME = re.compile(r"\bname is ([A-Z][\w'-]+)", re.I)


def name_from_memories(memories: list[str]) -> str | None:
    for m in memories:
        hit = _NAME.search(m)
        if hit:
            return hit.group(1).capitalize()
    return None


TOOL_VERBS = {
    "web_search": ("Searched the web", "Searching the web"),
    "calculator": ("Calculated", "Calculating"),
    "search_documents": ("Searched your documents", "Searching your documents"),
    "current_datetime": ("Checked the date", "Checking the date"),
    "get_stock_price": ("Looked up a stock price", "Looking up a stock price"),
    "get_weather": ("Checked the weather", "Checking the weather"),
}


def tool_done(name: str) -> str:
    return TOOL_VERBS.get(name, (f"Used {name}", ""))[0]


def tool_running(name: str) -> str:
    return TOOL_VERBS.get(name, ("", f"Using {name}"))[1] + "…"


def summarize_tools(names: list[str]) -> str:
    seen: list[str] = []
    for n in names:
        label = tool_done(n)
        if label not in seen:
            seen.append(label)
    if not seen:
        return ""
    text = ", ".join(seen[:-1]) + (" and " if len(seen) > 1 else "") + seen[-1]
    return text[0].upper() + text[1:]
