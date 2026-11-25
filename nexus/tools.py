"""Agent tools.

* web_search, calculator, get_stock_price    – chatbot-in-langgraph
* get_weather                                – langsmith-masterclass (ReAct agent)
* search_documents (Corrective RAG)          – corrective-rag + chatbot RAG tool
* MCP tools                                  – chatbot-in-langgraph MCP backend

Changes vs. the originals: API keys come from env (they were hard-coded),
key-dependent tools are only offered when a key exists, the calculator accepts
full expressions via a safe AST evaluator, and the document tool reads the
thread id from the run config instead of trusting the LLM to pass it.
"""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime

import requests
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from . import web
from .config import settings

HTTP_TIMEOUT = 15


@tool
def web_search(query: str) -> list[dict]:
    """Search the web for current events, facts or anything not in the conversation.
    Returns a list of results with title, url and content."""
    results = web.search(query, max_results=5)
    return results or [{"error": "No results (search provider unavailable)."}]


# --------------------------------------------------------------- calculator
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {name: getattr(math, name) for name in (
    "sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "asin", "acos", "atan",
    "floor", "ceil", "factorial", "radians", "degrees",
)} | {"abs": abs, "round": round, "min": min, "max": max}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"Unsupported expression element: {ast.dump(node)[:60]}")


def safe_eval(expression: str) -> float:
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    return _eval(ast.parse(expression, mode="eval"))


@tool
def calculator(expression: str) -> dict:
    """Evaluate an arithmetic expression exactly, e.g. "(12.5 * 3) / 7", "sqrt(2) + 2^10",
    "factorial(10)". Supports + - * / // % ** and common math functions. Always use this
    for arithmetic instead of computing in your head."""
    try:
        return {"expression": expression, "result": safe_eval(expression)}
    except ZeroDivisionError:
        return {"expression": expression, "error": "Division by zero is not allowed"}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


@tool
def current_datetime() -> str:
    """Return the current local date and time (useful for 'today', 'last week', ages, deadlines)."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M")


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock quote for a ticker symbol (e.g. 'AAPL', 'TSLA', 'RELIANCE.BSE')."""
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": settings.alphavantage_api_key},
            timeout=HTTP_TIMEOUT,
        )
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    try:
        r = requests.get(
            "http://api.weatherstack.com/current",
            params={"access_key": settings.weatherstack_api_key, "query": city},
            timeout=HTTP_TIMEOUT,
        )
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


@tool
def search_documents(query: str, config: RunnableConfig) -> dict:
    """Search the documents the user uploaded to this chat and the shared knowledge base
    using Corrective RAG (each passage is graded; weak results trigger a web search).
    Use for any question about the user's files, books or knowledge base."""
    from .rag.crag import run_crag

    configurable = (config or {}).get("configurable", {})
    result = run_crag(
        query,
        thread_id=configurable.get("thread_id"),
        include_knowledge=configurable.get("include_knowledge", True),
        with_generation=False,
        config={"callbacks": (config or {}).get("callbacks")},
    )
    return {
        "verdict": result.get("verdict"),
        "reason": result.get("reason"),
        "web_query": result.get("web_query"),
        "context": result.get("refined_context") or "(no relevant passages found)",
        "sources": result.get("sources", []),
    }


def builtin_tools() -> list[BaseTool]:
    tools: list[BaseTool] = [web_search, calculator, current_datetime, search_documents]
    if settings.alphavantage_api_key:
        tools.append(get_stock_price)
    if settings.weatherstack_api_key:
        tools.append(get_weather)
    return tools
