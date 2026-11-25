"""A dedicated background event loop (from chatbot-in-langgraph's MCP backend).

MCP tools are async-only; Streamlit and the rest of the graph are sync. Running
coroutines on one long-lived loop lets sync code call async tools safely.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine

_LOOP = asyncio.new_event_loop()
_THREAD = threading.Thread(target=_LOOP.run_forever, name="nexus-async", daemon=True)
_THREAD.start()


def run_async(coro: Coroutine[Any, Any, Any], timeout: float | None = 120) -> Any:
    return asyncio.run_coroutine_threadsafe(coro, _LOOP).result(timeout=timeout)
