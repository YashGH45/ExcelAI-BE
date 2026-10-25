# ExcelAI — Claude Code Context

## Project Overview
**ExcelAI** is a FastAPI backend that lets users upload Excel/CSV/PDF files and query them in natural language.
Queries are routed through intent classification → DuckDB SQL generation → LLM answer + optional Vega-Lite chart.
A "Smart Parser" mode collects user context about the dataset before generating richer metadata.

**Stack:** Python 3.11+, FastAPI, DuckDB, ChromaDB, Groq (chat/completions), local Ollama (conversation-memory embeddings only), rank-bm25, pdfplumber, pandas, openpyxl

---

## Running the Server
```bash
python main.py          # kills port 8000 if busy, then starts uvicorn on 127.0.0.1:8000
# OR
uvicorn app:app --reload --port 8000
```
Frontend dev server expected at `http://localhost:5173` (Vite).

---

## Environment Variables (`.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Required; all chat/completions go through Groq |
| `GROQ_CHAT_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Local embedding model, conversation-memory vector search only (`memory.py`) — not used for chat |
| `RETRIEVAL_TOP_K` | `5` | ChromaDB chunks per query |

---

## Key Directories
| Path | Contents |
|------|---------|
| `/` (root) | All Python source files (flat layout) |
| `uploads/` | User-uploaded files; auto-created |
| `vector_db/` | ChromaDB persistent store (metadata + conversation memory) |
| `graphify-out/` | `graph.json` — module dependency graph |
| `.codebase_context/` | Token-efficient context layer (INDEX.meta, per-file .meta, RESOLVE.md) |

---

## Entry Points
- **`main.py`** — process entry; start with `python main.py`
- **`app.py`** — FastAPI app with all routes

## API Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness check |
| POST | `/api/upload` | Upload Excel/CSV/PDF, generate metadata |
| POST | `/api/chat` | Natural language query |
| GET | `/api/download/{filename}` | Download processed file |
| GET | `/api/smart-parser/questions` | Get dataset context questions |
| POST | `/api/smart-parser/answers` | Submit answers, regenerate metadata |

---

## 5 Most Important Modules
1. **`app.py`** — API surface; start here to trace any request
2. **`sql_agent.py`** — NL→SQL pipeline, retry logic, fuzzy fallback (most complex logic)
3. **`metadata.py`** — ChromaDB RAG layer; column descriptions drive SQL quality
4. **`llm.py`** — Single LLM abstraction point; all chat/completions go through Groq
5. **`session.py`** — Per-user state (DuckDB conn, conversation memory, PDF index)

---

## Architecture Notes
- **No database server** — DuckDB runs in-memory per session; data lives in `uploads/`
- **ChromaDB** persists two collections per session: `metadata_<session_id>` and `memory_<session_id>`
- **Conversation memory** uses a rolling summary (max 5 recent turns in `deque`; older turns compressed by LLM). Turn embeddings for similarity retrieval still use a local Ollama model (`memory.py:_embed_for_memory`, `nomic-embed-text`) — this is independent of the chat backend and intentionally unchanged
- **PDF mode** uses BM25 (no embeddings) for page retrieval; stored in `Session.pdf_bm25`
- **All LLM prompts** live in `config.py` — edit there to tune behaviour without touching logic files

---

## Code Conventions
- Internal helpers are prefixed with `_` (e.g. `_generate_sql`, `_identify_dirty_rows`)
- All chat/completion calls go through `llm.get_response_from_llm()` — never call the Groq SDK directly elsewhere. The one exception is `memory.py`, which calls the local `ollama` package directly for conversation-memory embeddings (no Groq embeddings API exists)
- `dto_models.py` owns all cross-module data shapes; avoid inline dicts for API responses
- Column headers are normalised to snake_case at upload time (`file_loader._to_snake_case`)

---

## Dependencies (requirements.txt)
```
python-dotenv, fastapi, uvicorn, python-multipart, groq, openai,
ollama, chromadb, pandas, openpyxl, duckdb, psutil, pdfplumber, rank_bm25, altair
```

---

## Context Compression Layer
For token-efficient navigation in future sessions, read these instead of raw source:
- `.codebase_context/INDEX.meta` — one-liner per file
- `.codebase_context/metadata/<file>.meta` — detailed per-file summary
- `.codebase_context/RESOLVE.md` — symbol→file map + request flow diagrams
- `graphify-out/graph.json` — full dependency graph
