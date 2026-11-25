"""The master agent graph.

                ┌──────────► remember ──────────► END        (long-term memory extraction,
    START ──────┤                                             runs in parallel with the agent)
                └──────────► agent ◄────► tools              (tool loop: web, calculator, CRAG docs, MCP …)
                               │
                               ├── history too long ──► summarize ──► END
                               └──────────────────────────────────► END

State = messages (checkpointed per thread) + rolling summary.
Store = per-user long-term memories (shared across threads).
"""
from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from . import memory, threads
from .config import settings
from .llm import get_chat_model, get_utility_model
from .rag import indexing

BASE_PROMPT = """You are Nexus, a helpful, accurate assistant with tools and memory.
Today is {today}.

TOOLS
- Use `search_documents` for anything about the user's uploaded files or the knowledge base.
  It returns graded context plus a verdict; cite the sources you use like [filename p.N] or [title](url).
- Use `web_search` for current events or facts you are unsure of; cite URLs.
- Use `calculator` for any arithmetic. Never guess numbers.
- Other tools may be available; use them when they clearly help.
Do not invent tool results. If a tool fails, say so briefly and continue as best you can.
"""

MEMORY_BLOCK = """
USER MEMORY
You know these facts about the user from earlier conversations:
{memories}
Personalise naturally: address the user by name when known, reference their projects, tools
and preferences, and adapt tone and format to their stated preferences. Only use known facts;
never assume.
"""

SUMMARY_BLOCK = """
EARLIER IN THIS CONVERSATION (summary of messages no longer shown):
{summary}
"""

DOCS_BLOCK = """
DOCUMENTS AVAILABLE TO `search_documents`: {docs}
"""

FOLLOW_UP_BLOCK = """
At the end of each answer, suggest 3 short, relevant follow-up questions the user might ask next,
as a bulleted list under the heading "**You might also ask:**".
"""


class AgentState(MessagesState):
    summary: str


def _cfg(config: RunnableConfig) -> dict:
    return (config or {}).get("configurable", {})


def _last_human_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def build_system_prompt(state: AgentState, config: RunnableConfig, store: BaseStore | None) -> str:
    c = _cfg(config)
    parts = [BASE_PROMPT.format(today=datetime.now().strftime("%A, %d %B %Y"))]

    if c.get("persona"):
        parts.append(f"\nADDITIONAL INSTRUCTIONS FROM THE USER\n{c['persona']}\n")

    if store is not None and c.get("use_memory", settings.long_term_memory):
        mems = memory.recall(store, c.get("user_id", "default"), query=_last_human_text(state))
        if mems:
            parts.append(MEMORY_BLOCK.format(memories="\n".join(f"- {m}" for m in mems)))

    if state.get("summary"):
        parts.append(SUMMARY_BLOCK.format(summary=state["summary"]))

    docs = [d["filename"] for d in threads.thread_documents(c["thread_id"])] if c.get("thread_id") else []
    if c.get("include_knowledge", True):
        docs += [f"{s['file']} (knowledge base)" for s in indexing.knowledge_status() if s["indexed"]]
    if docs:
        parts.append(DOCS_BLOCK.format(docs=", ".join(docs)))

    if c.get("follow_ups", False):
        parts.append(FOLLOW_UP_BLOCK)
    return "".join(parts)


def build_agent(tools: list[BaseTool], checkpointer=None, store=None):
    model = get_chat_model()
    model_with_tools = model.bind_tools(tools) if tools else model

    def agent(state: AgentState, config: RunnableConfig, *, store: BaseStore):
        system = SystemMessage(content=build_system_prompt(state, config, store))
        history = memory.trim_for_model(state["messages"])
        response = model_with_tools.invoke([system, *history], config)
        return {"messages": [response]}

    def remember(state: AgentState, config: RunnableConfig, *, store: BaseStore):
        c = _cfg(config)
        if store is not None and c.get("use_memory", settings.long_term_memory):
            try:
                memory.extract_and_store(store, c.get("user_id", "default"), _last_human_text(state))
            except Exception:
                pass  # memory is best-effort; never break the chat turn
        return {}

    def summarize(state: AgentState):
        return memory.summarize_history(state["messages"], state.get("summary", ""), get_utility_model())

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        if len(state["messages"]) > settings.summarize_after:
            return "summarize"
        return END

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("remember", remember)
    g.add_node("summarize", summarize)
    g.add_edge(START, "agent")
    g.add_edge(START, "remember")
    g.add_edge("remember", END)

    if tools:
        g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        g.add_conditional_edges("agent", route_after_agent, ["tools", "summarize", END])
        g.add_edge("tools", "agent")
    else:
        g.add_conditional_edges("agent", route_after_agent, ["summarize", END])
    g.add_edge("summarize", END)

    return g.compile(checkpointer=checkpointer, store=store, name="nexus_agent")
