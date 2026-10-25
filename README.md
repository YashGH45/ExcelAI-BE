# ExcelAI

A FastAPI backend that lets users upload Excel/CSV/PDF files and query them in natural language. Queries are routed through intent classification → DuckDB SQL generation → LLM answer, with an optional Vega-Lite chart and cited source evidence. A "Smart Parser" mode collects context about the dataset from the user before generating richer column metadata.

## Stack

- **API:** FastAPI + Uvicorn
- **Query engine:** DuckDB (in-memory, per session — no database server)
- **Vector store:** ChromaDB (per-session metadata + conversation-memory collections)
- **Chat/completions:** Groq (`llama-3.3-70b-versatile` by default) — the only chat backend
- **Embeddings (conversation memory only):** local Ollama (`nomic-embed-text`)
- **File parsing:** pandas, openpyxl, pdfplumber
- **PDF retrieval:** rank-bm25 (keyword search, no embeddings)
- **Charts:** Vega-Lite v5 specs generated via LLM (altair-compatible)

> **Note on the LLM backend:** all chat/completion calls go through `llm.py::get_response_from_llm()`, which calls Groq exclusively — there is no local-LLM chat fallback. The local `ollama` package is still a dependency solely because `memory.py` uses it to embed conversation turns for similarity retrieval (Groq has no embeddings API). If the local Ollama server isn't running, conversation-memory similarity retrieval silently no-ops; it doesn't affect chat responses.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy `.env` and fill in your own Groq key:

```ini
GROQ_API_KEY=your_key_here
GROQ_CHAT_MODEL=llama-3.3-70b-versatile

# Local embeddings for conversation memory only (requires: ollama pull nomic-embed-text)
OLLAMA_EMBED_MODEL=nomic-embed-text

RETRIEVAL_TOP_K=5
```

## Running

```bash
python main.py
# kills anything already bound to port 8000, then starts uvicorn on 127.0.0.1:8000
```

or directly:

```bash
uvicorn app:app --reload --port 8000
```

The frontend dev server is expected at `http://localhost:5173` (Vite) — CORS is configured for that origin in `app.py`.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness check |
| POST | `/api/upload` | Upload Excel/CSV/PDF, generate metadata |
| POST | `/api/chat` | Natural language query |
| GET | `/api/download/{filename}` | Download a processed file |
| GET | `/api/smart-parser/questions` | Get dataset context questions (Smart Parser mode) |
| POST | `/api/smart-parser/answers` | Submit answers, regenerate metadata |

## Request Flow

**Upload** (`POST /api/upload`)
- PDF → extracted page-by-page, indexed with BM25 (`pdf_handler.py`)
- Excel/CSV → loaded into a per-session DuckDB table (`file_loader.py`); column headers normalized to snake_case
  - Normal mode: rich column metadata is generated and stored in ChromaDB immediately
  - Smart Parser mode: metadata generation is deferred until the client POSTs answers to `/api/smart-parser/answers`

**Chat** (`POST /api/chat`, `search.py`)
1. PDF sessions → BM25 keyword search + LLM answer (`pdf_handler.py`)
2. Excel/CSV sessions → intent classification (`intent.py`): `CLEANSE` routes to data cleaning (`cleanse.py`); everything else goes to the NL→SQL pipeline
3. **NL→SQL pipeline** (`sql_agent.py`), a 3-stage pipeline that scopes what gets sent to the LLM to keep prompts small:
   - **Stage A — Intent Detection** (`query_planner.py::detect_sql_intent`): classifies operations/target columns/filters from the query, or flags it as ambiguous
   - **Stage B — Query Planning** (`query_planner.py::build_query_plan`): turns the intent into a structured plan (select columns, aggregations, filters, group/order by) before SQL is generated; both stages degrade gracefully to `{}` on any LLM/parse failure so the pipeline still runs on the raw query
   - **Stage C — SQL Generation & Execution** (`sql_agent.py::_generate_sql` / `_execute_with_retry`): generates SQL against the DuckDB schema, executes with an agentic retry/self-heal loop, applies fuzzy-match fallback, and truncates oversized results (`MAX_RESULT_ROWS` token guard)
4. Response includes an optional Vega-Lite chart + insights (`charts.py`) and cited sources/evidence (`SpreadsheetSource` / `PDFSource` / `PDFEvidence` in `dto_models.py`) showing which rows or PDF passages backed the answer

**Conversation memory** (`memory.py`): a rolling window of the last 5 turns (`deque`), with older turns compressed into a running summary by the LLM. Turn embeddings for similarity retrieval use the local Ollama model noted above.

## Project Structure

| Path | Contents |
|------|----------|
| `app.py` | FastAPI app, all routes |
| `main.py` | Process entry point (`python main.py`) |
| `sql_agent.py` | NL→SQL pipeline: retry logic, fuzzy fallback |
| `query_planner.py` | Stage A/B of the SQL pipeline: intent detection + query planning |
| `metadata.py` | ChromaDB RAG layer; column descriptions drive SQL quality |
| `llm.py` | Single LLM abstraction point (Groq) |
| `memory.py` | Conversation memory: rolling summary + embedding-based retrieval |
| `session.py` | Per-user state: DuckDB connection, conversation memory, PDF index |
| `intent.py` | Query intent classifier (chart/cleanse/write/answer) |
| `search.py` | Top-level chat dispatcher |
| `charts.py` | Vega-Lite spec + insight generation |
| `cleanse.py` | Dirty-row detection and LLM-assisted repair |
| `file_loader.py` | Excel/CSV loading into pandas + DuckDB; header normalization |
| `pdf_handler.py` | BM25 PDF index and PDF Q&A |
| `dto_models.py` | All cross-module request/response/source data shapes |
| `config.py` | All LLM prompts, plus shared constants (e.g. `UPLOAD_DIR`) |
| `uploads/` | User-uploaded files (auto-created) |
| `vector_db/` | ChromaDB persistent store |

See also: [`CLAUDE.md`](CLAUDE.md) for AI-assistant-oriented project context, [`CHARTS.md`](CHARTS.md) for the charting pipeline, and [`PDF_SUPPORT.md`](PDF_SUPPORT.md) for the PDF/BM25 design rationale.

## Code Conventions

- Internal helpers are prefixed with `_` (e.g. `_generate_sql`, `_identify_dirty_rows`)
- All chat/completion calls go through `llm.get_response_from_llm()` — never call the Groq SDK directly elsewhere. The one exception is `memory.py`, which calls the local `ollama` package directly for embeddings
- `dto_models.py` owns all cross-module data shapes — avoid inline dicts for API responses
- Column headers are normalized to snake_case at upload time (`file_loader._to_snake_case`)
