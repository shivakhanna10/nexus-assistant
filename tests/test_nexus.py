"""End-to-end offline tests for the master project."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nexus import memory, service, threads
from nexus.config import settings
from nexus.persistence import get_store
from nexus.rag import crag, indexing
from nexus.tools import safe_eval
from nexus.workflows import evaluate_essay

TRANSFORMER_TXT = (
    "A transformer is a deep learning architecture based on self-attention. "
    "Attention mechanisms let the model weigh the importance of different tokens in the input. "
    "Unlike recurrent networks, transformers process sequences in parallel during training. "
) * 3
COOKING_TXT = "Pasta should be boiled in salted water for about ten minutes until al dente. " * 3


def collect(events):
    evs = list(events)
    errors = [e.text for e in evs if e.kind == "error"]
    assert not errors, errors
    return evs


def test_safe_calculator():
    assert safe_eval("2^10 + sqrt(16)") == 1028
    assert safe_eval("(12.5*3)/7") == 12.5 * 3 / 7
    for bad in ("__import__('os')", "open('x')", "2**100000"):
        try:
            safe_eval(bad)
            raise AssertionError(f"should reject {bad}")
        except Exception:
            pass


def test_chat_streams_and_persists():
    tid = threads.new_thread_id()
    evs = collect(service.stream_turn(tid, "alice", "hi there"))
    done = [e for e in evs if e.kind == "done"][0]
    assert done.text.startswith("Echo: hi there")
    history, _ = service.load_history(tid)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert threads.get_thread(tid).title != "New chat"  # auto-titled
    assert tid in {t.thread_id for t in threads.list_threads("alice")}


def test_tool_loop_calculator():
    tid = threads.new_thread_id()
    evs = collect(service.stream_turn(tid, "alice", "compute 6*7"))
    kinds = [e.kind for e in evs]
    assert "tool_start" in kinds and "tool_end" in kinds
    tool_end = [e for e in evs if e.kind == "tool_end"][0]
    assert "42" in tool_end.text
    assert "42" in [e for e in evs if e.kind == "done"][0].text
    history, _ = service.load_history(tid)
    assert history[-1]["tools"][0]["name"] == "calculator"


def test_long_term_memory_across_threads():
    user = "bob"
    collect(service.stream_turn(threads.new_thread_id(), user, "Hi, my name is Bob"))
    mems = [m["text"] for m in memory.list_memories(get_store(), user)]
    assert mems == ["User's name is Bob"]
    # Duplicate is not stored twice
    collect(service.stream_turn(threads.new_thread_id(), user, "my name is Bob"))
    assert len(memory.list_memories(get_store(), user)) == 1
    # New thread is personalised from memory
    evs = collect(service.stream_turn(threads.new_thread_id(), user, "hello again"))
    assert [e for e in evs if e.kind == "done"][0].text == "Hello Bob!"
    # Memory disabled -> not used
    evs = collect(service.stream_turn(threads.new_thread_id(), user, "hello", use_memory=False))
    assert [e for e in evs if e.kind == "done"][0].text.startswith("Echo")


def test_summarization_keeps_tool_pairs_intact():
    tid = threads.new_thread_id()
    for i in range(4):
        collect(service.stream_turn(tid, "carol", f"compute {i}+1"))
    state = service.get_app().graph.get_state({"configurable": {"thread_id": tid}}).values
    assert state.get("summary"), "summary should be created once history exceeds the limit"
    msgs = state["messages"]
    assert len(msgs) <= settings.summarize_after
    assert isinstance(msgs[0], HumanMessage), "kept history must start on a human turn"
    # every ToolMessage still has its AI tool call
    call_ids = {tc["id"] for m in msgs if isinstance(m, AIMessage) for tc in m.tool_calls}
    assert all(m.tool_call_id in call_ids for m in msgs if isinstance(m, ToolMessage))


def test_safe_cut_index():
    msgs = [HumanMessage("a", id="1"), AIMessage("", id="2", tool_calls=[{"name": "x", "args": {}, "id": "t"}]),
            ToolMessage("r", tool_call_id="t", id="3"), AIMessage("b", id="4"), HumanMessage("c", id="5"),
            AIMessage("d", id="6")]
    assert memory.safe_cut_index(msgs, keep_recent=3) == 4  # moves back from ToolMessage/AI to Human


def test_upload_index_cache_and_crag(tmp_path):
    tid = threads.new_thread_id()
    threads.upsert_thread(tid, "dave")
    info = indexing.ingest_upload(TRANSFORMER_TXT.encode(), "transformers.txt", tid)
    assert info.chunks >= 1 and not info.cached
    again = indexing.ingest_upload(TRANSFORMER_TXT.encode(), "transformers.txt", tid)
    assert again.cached and again.key == info.key  # fingerprint cache hit (rag_v4)

    res = crag.run_crag("What is a transformer?", thread_id=tid, include_knowledge=False)
    assert res["verdict"] == "CORRECT"
    assert res["kept_strips"] and all(("attention" in s.lower() or "transformer" in s.lower())
                                      for s in res["kept_strips"])
    assert res["answer"]
    assert res["sources"][0]["label"] == "transformers.txt"


def test_crag_uses_a_constant_number_of_model_calls(fake_models):
    """Grading and refinement are batched: calls don't grow with chunks or sentences."""
    tid = threads.new_thread_id()
    threads.upsert_thread(tid, "hana")
    indexing.ingest_upload((TRANSFORMER_TXT * 4).encode(), "long-notes.txt", tid)
    type(fake_models).calls.clear()
    res = crag.run_crag("What is a transformer?", thread_id=tid, include_knowledge=False, with_generation=False)
    assert len(res["docs"]) >= 2 and len(res["strips"]) >= 6
    assert type(fake_models).calls == ["ChunkGrades", "KeptSentences"]


def test_content_block_replies_are_shown_as_plain_text():
    """Providers that reply with content blocks must never leak raw block dicts to the UI."""
    tid = threads.new_thread_id()
    threads.upsert_thread(tid, "ivan")
    indexing.ingest_upload(TRANSFORMER_TXT.encode(), "blocks.txt", tid)
    assert crag.run_crag("What is a transformer?", thread_id=tid, include_knowledge=False)["answer"] == "Blocks answer."
    assert evaluate_essay("Short essay. " * 10)["overall_feedback"] == "Blocks answer."


def test_crag_incorrect_goes_to_web():
    tid = threads.new_thread_id()
    threads.upsert_thread(tid, "erin")
    indexing.ingest_upload(COOKING_TXT.encode(), "cooking.txt", tid)
    res = crag.run_crag("Explain attention", thread_id=tid, include_knowledge=False)
    assert res["verdict"] == "INCORRECT"
    assert res["web_query"] == "fake web query"
    assert res["web_docs"] and "Attention" in res["refined_context"]


def test_crag_no_docs_is_incorrect():
    res = crag.run_crag("anything", thread_id=None, include_knowledge=False, with_generation=False)
    assert res["verdict"] == "INCORRECT"


def test_document_tool_inside_chat_uses_thread_from_config():
    tid = threads.new_thread_id()
    threads.upsert_thread(tid, "frank")
    indexing.ingest_upload(TRANSFORMER_TXT.encode(), "t.txt", tid)
    evs = collect(service.stream_turn(tid, "frank", "docs: what is a transformer"))
    end = [e for e in evs if e.kind == "tool_end"][0]
    assert "CORRECT" in end.text and "t.txt" in end.text


def test_knowledge_base_indexing():
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    (settings.knowledge_dir / "kb.md").write_text(TRANSFORMER_TXT + " Knowledge base copy.", encoding="utf-8")
    assert not indexing.knowledge_status()[0]["indexed"]
    indexing.index_knowledge_base()
    assert indexing.knowledge_status()[0]["indexed"]
    hits = indexing.retrieve("transformer attention", thread_id=None, include_knowledge=True)
    assert hits and hits[0].metadata["source"] == "kb.md"
    (settings.knowledge_dir / "kb.md").unlink()


def test_delete_conversation():
    tid = threads.new_thread_id()
    collect(service.stream_turn(tid, "gina", "hello"))
    service.delete_conversation(tid)
    assert threads.get_thread(tid) is None
    assert service.load_history(tid)[0] == []


def test_essay_evaluator_parallel():
    r = evaluate_essay("India and AI. " * 20)
    assert r["individual_scores"] == [7, 7, 7]
    assert r["avg_score"] == 7.0 and r["overall_feedback"]
