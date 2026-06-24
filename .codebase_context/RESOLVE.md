# RESOLVE — IDE Index Stub

Quick navigation map for this codebase. Use these pointers before reading raw files.

## Entry Points
| File | Purpose |
|------|---------|
| `main.py` | Start server: `python main.py` |
| `app.py` | All FastAPI routes |

## Symbol → File Map
| Symbol | File |
|--------|------|
| `search()` | `search.py` |
| `text_to_sql_search()` | `sql_agent.py` |
| `get_response_from_llm()` | `llm.py` |
| `Session`, `SessionManager` | `session.py` |
| `generate_rich_metadata()` | `metadata.py` |
| `store_metadata_in_vectordb()` | `metadata.py` |
| `retrieve_relevant_metadata()` | `metadata.py` |
| `classify_intent()` | `intent.py` |
| `QueryIntent` | `intent.py` |
| `cleanse_data()` | `cleanse.py` |
| `store_conversation()` | `memory.py` |
| `generate_chart_and_insights()` | `charts.py` |
| `load_dataframe()`, `load_into_duckdb()` | `file_loader.py` |
| `pdf_search()`, `load_pdf_index()` | `pdf_handler.py` |
| `SearchResult`, `ChatRequest`, `UploadResponse` | `dto_models.py` |
| All LLM prompts | `config.py` |
| Keyword lists | `dictionary.py` |

## Request Flow (Excel mode)
```
POST /api/upload
  → load_dataframe() [file_loader.py]
  → load_into_duckdb() [file_loader.py]
  → generate_rich_metadata() [metadata.py]  ← LLM call
  → store_metadata_in_vectordb() [metadata.py]  ← ChromaDB

POST /api/chat
  → search() [search.py]
    → classify_intent() [intent.py]
    → text_to_sql_search() [sql_agent.py]
      → retrieve_relevant_metadata() [metadata.py]
      → _generate_sql() → DuckDB execute → retry loop
      → generate_chart_and_insights() [charts.py]  ← if chart intent
      → store_conversation() [memory.py]
```

## Request Flow (PDF mode)
```
POST /api/upload (pdf)
  → load_pdf_pages() [file_loader.py]
  → load_pdf_index() [pdf_handler.py]  ← BM25

POST /api/chat
  → search() → pdf_search() [pdf_handler.py]
    → BM25 score pages → LLM answer
```

## Key Directories
| Path | Contents |
|------|---------|
| `uploads/` | User-uploaded files (Excel, CSV, PDF) |
| `vector_db/` | ChromaDB persistent store (metadata + memory collections) |
| `graphify-out/` | graph.json — module dependency map |
| `.codebase_context/` | This context compression layer |

## LLM Backend Switch
Set `LLM_BACKEND=ollama` (default) or `LLM_BACKEND=groq` in `.env`.
All LLM calls funnel through `llm.get_response_from_llm()`.
