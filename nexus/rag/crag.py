"""Corrective RAG (CRAG) as a LangGraph sub-graph.

Faithful to the final ``corrective-rag/6_ambiguous`` notebook:

    retrieve -> evaluate each chunk (0..1 score)
       CORRECT    (any score > UPPER)   -> refine(internal docs)
       INCORRECT  (all scores < LOWER)  -> rewrite query -> web search -> refine(web docs)
       AMBIGUOUS  (otherwise)           -> rewrite query -> web search -> refine(internal + web)
    refine = decompose into sentence strips -> LLM keep/drop filter -> recompose
    -> generate (optional; the chat agent does its own generation)

Improvements: graders run in parallel (``.batch``), retrieval spans this chat's
uploads + the shared knowledge base, empty retrieval is treated as INCORRECT,
web fallback can be disabled, and sources are returned for citations.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .. import web
from ..config import settings
from ..llm import get_chat_model, get_utility_model
from .indexing import retrieve


class CragState(TypedDict, total=False):
    question: str
    thread_id: str | None
    include_knowledge: bool

    docs: list[Document]
    scores: list[float]
    good_docs: list[Document]
    verdict: str
    reason: str

    web_query: str
    web_docs: list[Document]

    strips: list[str]
    kept_strips: list[str]
    refined_context: str
    sources: list[dict]

    answer: str


# ------------------------------------------------------------------ prompts
class DocEvalScore(BaseModel):
    score: float = Field(description="Relevance in [0.0, 1.0]", ge=0.0, le=1.0)
    reason: str = Field(description="One short sentence")


class KeepOrDrop(BaseModel):
    keep: bool


class WebQuery(BaseModel):
    query: str


DOC_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict retrieval evaluator for RAG.\n"
     "You will be given ONE retrieved chunk and a question.\n"
     "Return a relevance score in [0.0, 1.0].\n"
     "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
     "- 0.0: chunk is irrelevant\n"
     "Be conservative with high scores. Also return a short reason."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])

FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance filter.\n"
     "Return keep=true only if the sentence directly helps answer the question.\n"
     "Use ONLY the sentence."),
    ("human", "Question: {question}\n\nSentence:\n{sentence}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user question into a web search query composed of keywords.\n"
     "Rules:\n"
     "- Keep it short (6-14 words).\n"
     "- If the question implies recency (e.g. recent/latest/last week/last month), "
     "add a constraint like (last 30 days).\n"
     "- Do NOT answer the question."),
    ("human", "Question: {question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer ONLY using the provided context.\n"
     "Cite sources inline like [source]. If the context is empty or insufficient, say: 'I don't know.'"),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


@lru_cache(maxsize=1)
def _chains():
    llm = get_utility_model()
    return (
        DOC_EVAL_PROMPT | llm.with_structured_output(DocEvalScore),
        FILTER_PROMPT | llm.with_structured_output(KeepOrDrop),
        REWRITE_PROMPT | llm.with_structured_output(WebQuery),
    )


def _batch_cfg() -> dict:
    return {"max_concurrency": settings.max_concurrency}


# -------------------------------------------------------------------- nodes
def retrieve_node(state: CragState) -> CragState:
    docs = retrieve(
        state["question"],
        thread_id=state.get("thread_id"),
        include_knowledge=state.get("include_knowledge", True),
    )
    return {"docs": docs}


def evaluate_node(state: CragState) -> CragState:
    docs = state.get("docs") or []
    if not docs:
        return {"scores": [], "good_docs": [], "verdict": "INCORRECT",
                "reason": "No indexed documents matched (or none are uploaded)."}

    doc_eval, _, _ = _chains()
    outs = doc_eval.batch(
        [{"question": state["question"], "chunk": d.page_content} for d in docs], config=_batch_cfg()
    )
    scores = [o.score for o in outs]
    lower, upper = settings.crag_lower_threshold, settings.crag_upper_threshold
    good = [d for d, s in zip(docs, scores) if s > lower]

    if any(s > upper for s in scores):
        return {"scores": scores, "good_docs": good, "verdict": "CORRECT",
                "reason": f"At least one retrieved chunk scored > {upper}."}
    if all(s < lower for s in scores):
        return {"scores": scores, "good_docs": [], "verdict": "INCORRECT",
                "reason": f"All retrieved chunks scored < {lower}."}
    return {"scores": scores, "good_docs": good, "verdict": "AMBIGUOUS",
            "reason": f"No chunk scored > {upper}, but not all were < {lower}."}


def rewrite_query_node(state: CragState) -> CragState:
    _, _, rewrite = _chains()
    return {"web_query": rewrite.invoke({"question": state["question"]}).query}


def web_search_node(state: CragState) -> CragState:
    q = state.get("web_query") or state["question"]
    web_docs = [
        Document(
            page_content=f"TITLE: {r['title']}\nURL: {r['url']}\nCONTENT:\n{r['content']}",
            metadata={"url": r["url"], "title": r["title"], "source": r["url"]},
        )
        for r in web.search(q)
    ]
    return {"web_docs": web_docs}


def decompose_to_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _docs_for_verdict(state: CragState) -> list[Document]:
    verdict = state.get("verdict")
    good, web_docs = state.get("good_docs") or [], state.get("web_docs") or []
    if verdict == "CORRECT":
        return good
    if verdict == "INCORRECT":
        return web_docs
    return good + web_docs


def _sources(docs: list[Document]) -> list[dict]:
    out, seen = [], set()
    for d in docs:
        m = d.metadata
        label = m.get("title") or m.get("source", "document")
        ref = m.get("url") or (f"p.{m['page'] + 1}" if isinstance(m.get("page"), int) else "")
        key = (label, ref)
        if key not in seen:
            seen.add(key)
            out.append({"label": label, "ref": ref, "url": m.get("url")})
    return out


def refine_node(state: CragState) -> CragState:
    docs = _docs_for_verdict(state)
    context = "\n\n".join(d.page_content for d in docs).strip()
    if not settings.crag_refine:
        return {"strips": [], "kept_strips": [], "refined_context": context, "sources": _sources(docs)}

    strips = decompose_to_sentences(context)
    kept: list[str] = []
    if strips:
        _, filt, _ = _chains()
        decisions = filt.batch(
            [{"question": state["question"], "sentence": s} for s in strips], config=_batch_cfg()
        )
        kept = [s for s, d in zip(strips, decisions) if d.keep]
    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": "\n".join(kept).strip(),
        "sources": _sources(docs),
    }


def generate_node(state: CragState) -> CragState:
    srcs = ", ".join(f"{s['label']} {s['ref']}".strip() for s in state.get("sources", []))
    context = state.get("refined_context", "")
    if srcs:
        context += f"\n\nSources: {srcs}"
    out = (ANSWER_PROMPT | get_chat_model(temperature=0)).invoke(
        {"question": state["question"], "context": context}
    )
    return {"answer": out.content}


def route_after_eval(state: CragState) -> str:
    if state["verdict"] == "CORRECT" or not settings.crag_web_fallback:
        return "refine"
    return "rewrite_query"


# -------------------------------------------------------------------- graph
def build_crag_graph(with_generation: bool = True):
    g = StateGraph(CragState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "evaluate")
    g.add_conditional_edges("evaluate", route_after_eval,
                            {"refine": "refine", "rewrite_query": "rewrite_query"})
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")

    if with_generation:
        g.add_node("generate", generate_node)
        g.add_edge("refine", "generate")
        g.add_edge("generate", END)
    else:
        g.add_edge("refine", END)
    return g.compile()


@lru_cache(maxsize=2)
def get_crag_graph(with_generation: bool = True):
    return build_crag_graph(with_generation)


def run_crag(question: str, thread_id: str | None = None, include_knowledge: bool = True,
             with_generation: bool = True, config: dict | None = None) -> CragState:
    return get_crag_graph(with_generation).invoke(
        {"question": question, "thread_id": thread_id, "include_knowledge": include_knowledge},
        config={**(config or {}), "run_name": "corrective_rag", "tags": ["crag"]},
    )
