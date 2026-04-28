# PDF Support — VectorLess PageIndex

## Overview

ExcelAI now supports PDF files alongside Excel/CSV. The implementation deliberately avoids neural embeddings and vector databases for PDF retrieval. Instead it uses a **BM25 PageIndex** — the same ranking algorithm that powers Elasticsearch and Apache Solr — built directly over the PDF's pages at upload time.

---

## Why VectorLess?

The existing Excel/CSV path already uses ChromaDB + Ollama embeddings for conversation memory. Adding another vector collection for PDF content would introduce:

- Ollama as a hard dependency (must be running locally)
- Slow indexing time on large PDFs
- Non-deterministic results from approximate nearest-neighbour search

BM25 solves all three: it is pure Python, indexes instantly, and produces deterministic, explainable scores based on term frequency and inverse document frequency.

---

## Dependencies

Two packages were added:

| Package | Version | Purpose |
|---------|---------|---------|
| `pdfplumber` | `0.11.x` | PDF text extraction, page-by-page |
| `rank_bm25` | `0.2.x` | BM25Okapi scoring (no embeddings) |

Both are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

### Transitive dependencies pulled in by `pdfplumber`

| Package | Role |
|---------|------|
| `pdfminer.six` | Low-level PDF parsing |
| `pypdfium2` | Fast PDF rendering (used for image extraction) |
| `cryptography` | Handles encrypted PDFs |

---

## Scope

### In scope

- Text-based PDFs (reports, contracts, research papers, manuals, invoices, etc.)
- Multi-page PDFs of any length
- Follow-up / conversational queries ("what else does it say about X?", "expand on that")
- Seamless switching between PDF and Excel/CSV sessions — uploading a new file of either type resets all state

### Out of scope

- **Scanned / image-only PDFs** — pdfplumber extracts text layer only; OCR is not included. Uploading a scanned PDF returns a `422` error with a clear message.
- **Tables inside PDFs** — table cells may be extracted as raw text with inconsistent spacing. Structured table queries should use Excel/CSV instead.
- **Password-protected PDFs** — not handled; will raise an exception at upload.
- **Multi-file sessions** — only one document (PDF or spreadsheet) can be active at a time.

---

## Architecture

### Component map

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Vite, localhost:5173)                        │
└────────────────────┬────────────────────────────────────┘
                     │  POST /api/upload  (file)
                     │  POST /api/chat    (message)
                     ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI  ·  app.py                                     │
│                                                         │
│  /api/upload                                            │
│   ├── .pdf  ──► load_pdf_pages()  ──► load_pdf_index() │
│   │             set_mode("pdf")                         │
│   └── .xlsx/.csv ──► load_dataframe() ──► DuckDB       │
│                       set_mode("excel")                 │
│                                                         │
│  /api/chat  ──► search(query)                           │
└────────────────────┬────────────────────────────────────┘
                     │
          _current_mode == "pdf" ?
                 │           │
               YES           NO
                │             │
                ▼             ▼
        _pdf_search()   _text_to_sql_search()
                │
    BM25 score all pages
                │
       top-K pages (score > 0)
                │
    build context string
                │
    inject conversation history
                │
    Groq LLM → natural language answer
                │
    _store_pdf_conversation()
    (updates _recent_pdf_turns deque)
```

---

## How It Works — Step by Step

### 1. Upload (`POST /api/upload`)

```
User uploads document.pdf
        │
        ▼
app.py  detects filename ends with ".pdf"
        │
        ▼
services.load_pdf_pages(filepath)
  └── pdfplumber.open(pdf)
        for each page:
          text = page.extract_text()   # pure text layer, no OCR
          if text:
            pages.append({page_num: N, text: "..."})
        │
        ▼ list[{page_num, text}]  (blank pages silently skipped)
        │
        ▼
ai_client.load_pdf_index(pages)
  └── tokenize each page:  page["text"].lower().split()
      _pdf_bm25 = BM25Okapi(tokenized_corpus)   # built once, reused for all queries
      _pdf_pages = pages                          # kept in memory for context assembly
        │
        ▼
ai_client.set_mode("pdf")       # routes future chat queries to _pdf_search()
        │
        ▼
UploadResponse(filename, rows=len(pages), col_count=0, columns=[], file_type="pdf")
```

The BM25 index is built **once at upload time** and held in memory. There is no re-indexing per query.

---

### 2. Query (`POST /api/chat`)

```
User: "What are the payment terms?"
        │
        ▼
search(query)
  └── _current_mode == "pdf"  →  _pdf_search(query, top_k=3)
        │
        ▼
  tokens = query.lower().split()          # ["what", "are", "the", "payment", "terms"]
  scores = _pdf_bm25.get_scores(tokens)   # BM25 score for every page
        │
        ▼
  top_indices = sorted by score, descending, take first top_k
  filter: only pages where score > 0      # skip pages with zero relevance
        │
        ▼
  context = "[Page 4]\n<text>\n---\n[Page 7]\n<text>\n---\n[Page 12]\n<text>"
        │
        ▼
  inject _recent_pdf_turns (last 5 Q/Pages/A tuples)
  for follow-up resolution
        │
        ▼
  Groq LLM  ←  PDF_ANSWER_PROMPT.format(context, history, query)
        │
        ▼
  answer = "According to page 4, payment is due within 30 days of invoice..."
        │
        ▼
  _store_pdf_conversation(query, "p.4, p.7, p.12", answer)
  → appended to _recent_pdf_turns deque (maxlen=5)
        │
        ▼
  return [answer]
```

---

### 3. Follow-up queries

The `_recent_pdf_turns` deque (capped at 5) is formatted into a conversation history block and injected into every subsequent prompt:

```
[Turn 1]
Q: What are the payment terms?
Pages: p.4, p.7
A: Payment is due within 30 days of invoice date...

[MOST RECENT]
Q: What happens if payment is late?
Pages: p.7, p.9
A: A 1.5% monthly late fee applies after the grace period...
```

The LLM uses this history to resolve pronouns and contextual references without needing a fresh retrieval pass for every turn.

---

## BM25 — Algorithm Explained

BM25 (Best Match 25) is a probabilistic ranking function. For a query term `t` and page `d`:

```
score(t, d) = IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))
```

| Symbol | Meaning |
|--------|---------|
| `tf(t,d)` | How many times term `t` appears in page `d` |
| `IDF(t)` | Inverse document frequency — rare terms score higher |
| `\|d\|` | Length of page `d` in tokens |
| `avgdl` | Average page length across the whole PDF |
| `k1`, `b` | Tuning constants (BM25Okapi defaults: k1=1.5, b=0.75) |

**In plain English:** a page scores high if it contains the query's rare terms many times, but the score is penalised if the page is unusually long (so short, dense pages are not disadvantaged against verbose ones).

---

## State Management

All PDF state lives as module-level globals in `ai_client.py`:

| Variable | Type | Purpose |
|----------|------|---------|
| `_current_mode` | `str` | `"excel"` or `"pdf"` — routes `search()` calls |
| `_pdf_pages` | `list[dict] \| None` | Raw page data from pdfplumber |
| `_pdf_bm25` | `BM25Okapi \| None` | The live BM25 index |
| `_recent_pdf_turns` | `deque(maxlen=5)` | Last 5 `(query, page_refs, answer)` tuples |

### Session reset

`delete()` (called by the upload endpoint on every new upload) clears all four variables and resets `_current_mode` back to `"excel"`. This guarantees:

- No state bleeds from a previous PDF into a new session
- Uploading an Excel file after a PDF session cleanly restores the Text-to-SQL path

---

## File Changes Summary

| File | Change |
|------|--------|
| `requirements.txt` | Added `pdfplumber`, `rank_bm25` |
| `services.py` | Added `load_pdf_pages(filepath) → list[dict]` |
| `ai_client.py` | `BM25Okapi` import; PDF state globals; `set_mode()`, `load_pdf_index()`, `_pdf_search()`, `_store_pdf_conversation()`, `_format_recent_pdf_turns()`; updated `search()` and `delete()` |
| `app.py` | Upload endpoint branches on `.pdf` extension; imports `load_pdf_index`, `set_mode`, `load_pdf_pages` |
| `config.py` | Added `PDF_ANSWER_PROMPT` |
| `dto_models.py` | Added `file_type: str = "excel"` field to `UploadResponse` |

---

## UploadResponse — PDF vs Excel

When a PDF is uploaded the response shape is:

```json
{
  "filename": "contract.pdf",
  "rows": 18,
  "col_count": 0,
  "columns": [],
  "file_type": "pdf"
}
```

`rows` is the number of **text-bearing pages** (blank pages are skipped). `col_count` and `columns` are empty — they are meaningful only for spreadsheet uploads.

---

## Known Limitations

| Limitation | Reason | Workaround |
|------------|--------|------------|
| Scanned PDFs not supported | pdfplumber reads text layer only, no OCR | Convert to text PDF using Adobe Acrobat or Tesseract first |
| Very large PDFs may use significant memory | All page texts held in RAM | No mitigation for now; tested up to ~300 pages |
| Token limit on dense pages | A single page with thousands of words may overflow Groq's context | top_k is capped at 3; reduce if hitting token limits |
| BM25 is keyword-only | Cannot match synonyms or paraphrases ("payment" ≠ "remittance") | For semantic queries, consider re-enabling the vector search path |
| No table structure awareness | PDF tables are extracted as plain text | Use Excel/CSV for structured tabular data |

---

## Future Considerations

- **Hybrid retrieval** — combine BM25 with a lightweight sentence-transformer re-ranker for synonym-aware retrieval, still without a full vector DB
- **OCR integration** — add `pytesseract` + `pdf2image` as an optional fallback for scanned PDFs
- **Chunking strategy** — split pages that exceed a token threshold into overlapping sub-chunks before indexing
- **Multi-document sessions** — maintain separate PageIndex instances per document and let the user switch between them