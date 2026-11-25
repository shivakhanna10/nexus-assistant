"""Terminal interface.

    python cli.py chat  [--user NAME] [--thread ID]   interactive streaming chat
    python cli.py index [--rebuild]                    index files in data/knowledge
    python cli.py crag  "question"                     run Corrective RAG and show its decisions
    python cli.py essay path/to/essay.txt              parallel essay evaluation
    python cli.py report "topic"                       report -> 5-point summary chain
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def cmd_chat(args):
    from nexus import service, threads

    thread_id = args.thread or threads.new_thread_id()
    print(f"Nexus chat · user={args.user} · thread={thread_id}\nType /exit to quit, /new for a new thread.\n")
    while True:
        try:
            text = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        if text == "/new":
            thread_id = threads.new_thread_id()
            print(f"(new thread {thread_id})")
            continue
        print("nexus › ", end="", flush=True)
        for ev in service.stream_turn(thread_id, args.user, text, follow_ups=False):
            if ev.kind == "token":
                print(ev.text, end="", flush=True)
            elif ev.kind == "tool_start":
                print(f"\n  [tool] {ev.name}({ev.data})", flush=True)
                print("nexus › ", end="", flush=True)
            elif ev.kind == "error":
                print(f"\n  [error] {ev.text}")
        print("\n")


def cmd_index(args):
    from nexus.rag import indexing

    files = indexing.knowledge_files()
    if not files:
        print(f"No files in {indexing.settings.knowledge_dir}")
        return
    for info in indexing.index_knowledge_base(
        force_rebuild=args.rebuild, progress=lambda i, n, name: print(f"[{i}/{n}] {name} …", flush=True)
    ):
        print(f"   {info.filename}: {info.pages} pages, {info.chunks} chunks {'(cached)' if info.cached else ''}")


def cmd_crag(args):
    from nexus.rag.crag import run_crag

    r = run_crag(args.question, include_knowledge=True)
    print("VERDICT:", r.get("verdict"))
    print("REASON:", r.get("reason"))
    if r.get("web_query"):
        print("WEB QUERY:", r["web_query"])
    print("SCORES:", [round(s, 2) for s in r.get("scores", [])])
    print(f"KEPT {len(r.get('kept_strips', []))}/{len(r.get('strips', []))} sentences")
    print("\nANSWER:\n", r.get("answer"))


def cmd_essay(args):
    from nexus.workflows import evaluate_essay

    r = evaluate_essay(Path(args.path).read_text(encoding="utf-8"))
    for k in ("language_feedback", "analysis_feedback", "clarity_feedback", "overall_feedback"):
        print(f"\n== {k.replace('_', ' ').title()} ==\n{r.get(k, '')}")
    print("\nScores:", r.get("individual_scores"), "Average:", round(r.get("avg_score", 0), 2))


def cmd_report(args):
    from nexus.workflows import report_and_summary

    r = report_and_summary(args.topic)
    print(r["summary"])


def main():
    p = argparse.ArgumentParser(description="Nexus assistant CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat"); c.add_argument("--user", default="default"); c.add_argument("--thread")
    c.set_defaults(fn=cmd_chat)
    i = sub.add_parser("index"); i.add_argument("--rebuild", action="store_true"); i.set_defaults(fn=cmd_index)
    q = sub.add_parser("crag"); q.add_argument("question"); q.set_defaults(fn=cmd_crag)
    e = sub.add_parser("essay"); e.add_argument("path"); e.set_defaults(fn=cmd_essay)
    r = sub.add_parser("report"); r.add_argument("topic"); r.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
