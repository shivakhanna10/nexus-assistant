"""Offline test harness: a scripted fake chat model + deterministic embeddings.

No API keys or network needed. The fake model understands tool binding and
structured output, so the real graphs (agent, CRAG, memory, essay) run end to end.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, ClassVar

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="nexus-test-")
os.environ["DATA_DIR"] = _TMP
os.environ["KNOWLEDGE_DIR"] = str(Path(_TMP) / "knowledge")
os.environ["MCP_CONFIG"] = str(Path(_TMP) / "no-mcp.json")
os.environ["OPENAI_API_KEY"] = "sk-test"
os.environ["SUMMARIZE_AFTER_MESSAGES"] = "8"
os.environ["KEEP_RECENT_MESSAGES"] = "2"
os.environ.pop("TAVILY_API_KEY", None)
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402


class FakeChatModel(BaseChatModel):
    """Scripted model: answers structured-output requests with plausible JSON,
    calls `calculator` when asked to compute, and summarises tool results."""

    @property
    def _llm_type(self) -> str:
        return "fake-scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    # -- structured outputs keyed by schema name
    def _structured(self, name: str, text: str) -> dict:
        if name == "ChunkGrades":
            chunks = re.findall(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\Z)", text.split("Chunks:", 1)[-1], re.S)
            return {"grades": [{"score": 0.9 if "transformer" in c.lower() else 0.1, "reason": "fake"}
                               for _, c in chunks]}
        if name == "KeptSentences":
            sents = re.findall(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\Z)", text.split("Sentences:", 1)[-1], re.S)
            return {"keep": [int(i) for i, s in sents
                             if "attention" in s.lower() or "transformer" in s.lower()]}
        if name == "DocEvalScore":
            score = 0.9 if "transformer" in text.lower() else 0.1
            return {"score": score, "reason": "fake grader"}
        if name == "KeepOrDrop":
            return {"keep": "attention" in text.lower() or "transformer" in text.lower()}
        if name == "WebQuery":
            return {"query": "fake web query"}
        if name == "MemoryDecision":
            m = re.search(r"my name is (\w+)", text, re.I)
            if m:
                return {"should_write": True,
                        "memories": [{"text": f"User's name is {m.group(1)}", "is_new": True}]}
            return {"should_write": False, "memories": []}
        if name == "EvaluationSchema":
            return {"feedback": "Solid structure.", "score": 7}
        return {}

    calls: ClassVar[list] = []  # schema/tool name of every call, for call-count assertions

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        tools = kwargs.get("tools") or []
        FakeChatModel.calls.append(tools[0]["function"]["name"] if tools and kwargs.get("tool_choice") else "chat")
        tool_choice = kwargs.get("tool_choice")
        text = "\n".join(m.content if isinstance(m.content, str) else str(m.content) for m in messages)
        last = messages[-1]

        if tools and tool_choice:  # with_structured_output
            name = tools[0]["function"]["name"]
            msg = AIMessage(content="", tool_calls=[{
                "name": name, "args": self._structured(name, text), "id": f"call_{uuid.uuid4().hex[:8]}"}])
        elif tools and isinstance(last, HumanMessage) and last.content.lower().startswith("compute"):
            expr = last.content.split(" ", 1)[1]
            msg = AIMessage(content="", tool_calls=[{
                "name": "calculator", "args": {"expression": expr}, "id": f"call_{uuid.uuid4().hex[:8]}"}])
        elif tools and isinstance(last, HumanMessage) and last.content.lower().startswith("docs:"):
            msg = AIMessage(content="", tool_calls=[{
                "name": "search_documents", "args": {"query": last.content[5:].strip()},
                "id": f"call_{uuid.uuid4().hex[:8]}"}])
        elif "Answer ONLY using the provided context" in text or "Based on the following feedback" in text:
            # Gemini 3 / Anthropic style: a list of content blocks, not a string.
            msg = AIMessage(content=[{"type": "text", "text": "Blocks answer.", "extras": {"signature": "abc"}}])
        elif isinstance(last, ToolMessage):
            msg = AIMessage(content=f"Tool said: {last.content}")
        else:
            system = messages[0].content if messages and messages[0].type == "system" else ""
            name = re.search(r"User's name is (\w+)", system)
            msg = AIMessage(content=f"Hello {name.group(1)}!" if name else f"Echo: {last.content}"[:200])
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    import nexus.llm as llm
    from nexus.rag import indexing

    fake = FakeChatModel()
    emb = DeterministicFakeEmbedding(size=64)
    monkeypatch.setattr(llm, "init_chat_model", lambda *a, **k: fake)
    monkeypatch.setattr(llm, "get_embeddings", lambda *a, **k: emb)
    monkeypatch.setattr(indexing, "get_embeddings", lambda *a, **k: emb)
    llm.get_chat_model.cache_clear()
    llm.get_utility_model.cache_clear()
    for mod in ("nexus.rag.crag", "nexus.memory"):
        m = sys.modules.get(mod)
        for attr in ("_chains", "_batch_chains", "_extractor"):
            if m is not None and hasattr(getattr(m, attr, None), "cache_clear"):
                getattr(m, attr).cache_clear()
    import nexus.service as service
    service._APP = None
    # No network in tests.
    import nexus.web as web
    monkeypatch.setattr(web, "search", lambda q, max_results=5: [
        {"title": "Fake result", "url": "https://example.com", "content": "Attention mechanisms weigh tokens."}
    ])
    yield fake
