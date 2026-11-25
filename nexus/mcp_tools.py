"""Model Context Protocol tools (from chatbot-in-langgraph / langgraph_mcp_backend).

Servers are configured in ``mcp_servers.json`` (same schema as
``MultiServerMCPClient``) instead of being hard-coded. Async MCP tools are wrapped
as sync tools that execute on the shared background loop, so the main graph,
the SQLite checkpointer and Streamlit all stay synchronous.
"""
from __future__ import annotations

import json
import logging

from langchain_core.tools import BaseTool, StructuredTool

from .async_runner import run_async
from .config import settings

log = logging.getLogger(__name__)


def load_server_config() -> dict:
    path = settings.mcp_config_path
    if not path.exists():
        return {}
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return cfg.get("mcpServers", cfg)
    except Exception as exc:
        log.warning("Could not parse %s: %s", path, exc)
        return {}


def _syncify(tool: BaseTool) -> BaseTool:
    def _call(**kwargs):
        try:
            return run_async(tool.ainvoke(kwargs))
        except Exception as exc:
            return f"MCP tool '{tool.name}' failed: {exc}"

    return StructuredTool.from_function(
        func=_call,
        name=tool.name,
        description=tool.description or tool.name,
        args_schema=tool.args_schema,
        metadata={"mcp": True},
    )


def load_mcp_tools() -> tuple[list[BaseTool], list[str]]:
    """Return (tools, errors). Never raises – a broken server just contributes no tools."""
    servers = load_server_config()
    if not servers:
        return [], []
    tools: list[BaseTool] = []
    errors: list[str] = []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        return [], ["langchain-mcp-adapters is not installed"]

    # Load each server separately so one failure does not hide the others.
    for name, spec in servers.items():
        try:
            client = MultiServerMCPClient({name: spec})
            tools += [_syncify(t) for t in run_async(client.get_tools(), timeout=60)]
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            log.warning("MCP server %s unavailable: %s", name, exc)
    return tools, errors
