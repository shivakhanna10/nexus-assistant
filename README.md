# 🧭 Nexus: the master LangGraph assistant

One app that combines the best features of the four projects in `D:\ChatBot`:

| Source project | Feature carried over | Where it lives |
|---|---|---|
| **chatbot-in-langgraph** | Streamlit chat with token streaming | `ui/chat.py`, `nexus/service.py` |
| | Multiple conversations: new chat, resume, list | `nexus/threads.py` (now with auto titles + delete) |
| | SQLite checkpointer persistence | `nexus/persistence.py` |
| | Tool calling: web search, calculator, stock price | `nexus/tools.py` |
| | MCP tools (stdio + streamable HTTP) on a background event loop | `nexus/mcp_tools.py`, `nexus/async_runner.py` |
| | Ask questions about a PDF uploaded to a chat (RAG) | `nexus/rag/indexing.py` (indexes now survive restarts) |
| | Live "🔧 Using tool…" status in the UI | `ui/chat.py` |
| **corrective-rag** | Full CRAG: grade each chunk → CORRECT / INCORRECT / AMBIGUOUS | `nexus/rag/crag.py` |
| | Query rewrite → web search fallback (Tavily) | `nexus/rag/crag.py`, `nexus/web.py` |
| | Sentence-level refinement (split → filter → recombine) | `nexus/rag/crag.py` |
| | Multi-book knowledge base | **Knowledge Base** page |
| | Step-by-step pipeline view | **CRAG Inspector** page |
| **langsmith-masterclass** | LangSmith tracing with run names, tags and metadata | everywhere (`@traceable`, run configs) |
| | Disk index cache keyed by file fingerprint + settings (rag_v4) | `nexus/rag/indexing.py` |
| | Weather tool (from the ReAct agent) | `nexus/tools.py` |
| | Parallel essay evaluator (fan-out/fan-in + structured output) | **Workflows** page, `nexus/workflows.py` |
| | Sequential report → summary chain | **Workflows** page |
| **memory-in-llms** | Short-term memory per thread (checkpointer) | `nexus/persistence.py` |
| | Trimming history to a token budget | `nexus/memory.py` |
| | Rolling summary + deleting old messages | `nexus/memory.py` (never separates a tool call from its result) |
| | Long-term memory: extraction with duplicate detection, personalized prompt | `nexus/memory.py`, `nexus/agent.py` |
| | Semantic memory search (embeddings) | `SEMANTIC_MEMORY=true` |
| | Postgres checkpointer + store | `POSTGRES_URI`, `docker-compose.yml` |

### Improvements over the originals
- **No hard-coded secrets.** The Alpha Vantage and Weatherstack keys that were in the source code now come from `.env`. *Rotate the keys that were committed in the old projects.*
- **Switch model provider with an env var** (`init_chat_model`): OpenAI by default, Anthropic, etc. by changing `CHAT_MODEL`.
- **Document tool reads the chat id from the run config** instead of trusting the LLM to pass `thread_id`.
- **Uploaded files are cached and persisted.** Re-uploading the same file is instant and documents stay attached after a restart.
- **CRAG graders run in parallel** (`.batch`), so it's much faster than the one-call-at-a-time loops in the notebooks.
- **Safe calculator** that evaluates full expressions (`sqrt(2)*10^3`) via the AST, with no `eval`.
- **Long-term memory runs in parallel** with the answer, so it adds no extra wait.
- **Memory manager UI**: view, add, delete and clear memories per user profile.
- **Graceful degradation**: missing keys hide tools, a broken MCP server doesn't break the others, and DuckDuckGo is used if Tavily isn't configured.
- An **offline test suite** (fake LLM + fake embeddings) covers the agent, tools, memory, summarization, CRAG and workflows.

## Architecture

```
                 ┌─► remember (extract long-term memories) ─► END
   START ────────┤
                 └─► agent ◄──► tools [web_search · calculator · datetime · search_documents(CRAG) · stock · weather · MCP…]
                       │
                       ├─ history > N messages ─► summarize ─► END
                       └─────────────────────────────────────► END

   search_documents ─► CRAG subgraph:
      retrieve (chat uploads + knowledge base) → grade chunks → CORRECT ─────────────────────► refine → (answer)
                                                              → INCORRECT/AMBIGUOUS → rewrite → web search ─┘
```

```
master-chatbot/
├── app.py                 # Streamlit entry (multipage navigation)
├── cli.py                 # terminal chat / index / crag / essay / report
├── nexus/
│   ├── config.py          # every setting, from .env
│   ├── llm.py             # chat / utility / embedding model factories
│   ├── persistence.py     # SQLite or Postgres checkpointer + store
│   ├── threads.py         # conversation registry (titles, docs per chat)
│   ├── agent.py           # main LangGraph agent
│   ├── memory.py          # trimming, summarization, long-term memory
│   ├── tools.py           # built-in tools
│   ├── mcp_tools.py       # MCP loader (sync wrappers)
│   ├── web.py             # Tavily / DuckDuckGo
│   ├── workflows.py       # essay evaluator, report chain
│   ├── service.py         # streaming events, history, status
│   └── rag/
│       ├── indexing.py    # load → split → embed → FAISS, fingerprint cache
│       └── crag.py        # Corrective RAG graph
├── ui/                    # Streamlit pages
└── tests/                 # offline tests
```

## Quick start (Windows PowerShell)

```powershell
cd D:\ChatBot\master-chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then set OPENAI_API_KEY or GOOGLE_API_KEY
streamlit run app.py
```

**Using Gemini (free tier):** get a key at aistudio.google.com and put it in `.env`:
```
GOOGLE_API_KEY=...
CHAT_MODEL=google_genai:gemini-3.5-flash
UTILITY_MODEL=google_genai:gemini-3.5-flash
EMBEDDING_MODEL=google_genai:gemini-embedding-001
MAX_CONCURRENCY=2
```

Optional:
- **Knowledge base:** copy PDFs (for example `..\corrective-rag-main\documents\*.pdf` or `..\langsmith-masterclass-main\islr.pdf`) into `data\knowledge\`, then click **Index now** on the Knowledge Base page or run `python cli.py index`. Large books cost embedding tokens once; after that they load from cache.
- **MCP:** copy `mcp_servers.example.json` to `mcp_servers.json` and edit it.
- **Postgres:** run `docker compose up -d` and set `POSTGRES_URI` in `.env`.
- **LangSmith:** set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`.

### CLI
```powershell
python cli.py chat --user om
python cli.py crag "Batch normalization vs layer normalization"
python cli.py essay my_essay.txt
python cli.py report "Unemployment in India"
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a **private** GitHub repository (it must be the repository root, so
   `.streamlit/config.toml` is picked up).
2. On share.streamlit.io choose **Create app**, pick the repo and branch, set the main file to `app.py`,
   and under **Advanced settings** choose Python 3.11.
3. Paste your secrets (see `.streamlit/secrets.example.toml`) into **Secrets**.
4. Deploy. The first build takes a few minutes.

What to know about hosting there:
- **The disk is wiped on every restart and after the app sleeps.** Set `POSTGRES_URI` (a free
  database from neon.tech works) so conversations, the conversation list and memories persist.
  Uploaded documents and the knowledge base are indexed on disk, so re-upload them after a restart.
- **Keep it private.** There are no user accounts, and everyone shares the profiles. Either keep the
  app private in Streamlit's sharing settings (invite viewers by email) or set `APP_PASSWORD`.
- **Web search:** DuckDuckGo usually blocks cloud servers, so set `TAVILY_API_KEY` (free tier).
- **Resources are limited.** Avoid indexing very large PDFs there; index them locally instead.
- MCP servers that run as local processes (`stdio`) won't work there; HTTP MCP servers do.

### Tests (no API key needed)
```powershell
python -m pytest -q tests
```
