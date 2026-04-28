import os
import json
import duckdb
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
from dotenv import load_dotenv
from groq import Groq
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
import ollama
import chromadb
from rank_bm25 import BM25Okapi

load_dotenv()

USE_VECTOR_SEARCH = False

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
groq_chat_model = os.environ["GROQ_CHAT_MODEL"]
groq_mini_chat_model = os.environ["GROQ_MINI_CHAT_MODEL"]
gemma_model = os.environ["GEMMA_MODEL"]

# ── ChromaDB — always-on for conversation memory ─────────────────────────────
_chroma_client = chromadb.PersistentClient(path="./vector_db")
_conv_collection = _chroma_client.get_or_create_collection(name="conversation_memory")

if USE_VECTOR_SEARCH:
    collection = _chroma_client.get_or_create_collection(name="my_collection")

_duck_conn: duckdb.DuckDBPyConnection | None = None
_dataset_summary: str | None = None

# ── PDF PageIndex state (VectorLess BM25 retrieval) ──────────────────────────
_current_mode: str = "excel"
_pdf_pages: list[dict] | None = None
_pdf_bm25: BM25Okapi | None = None
_recent_pdf_turns: deque = deque(maxlen=5)

# ── Sequential turn buffer — primary context for follow-up resolution ────────
# Stores (query, sql, answer) tuples in order; capped at 5 most recent turns.
_recent_turns: deque = deque(maxlen=5)

_SUMMARY_KEYWORDS = {
    "summary", "summarize", "summarise", "summarization",
    "overview", "describe the data", "describe the dataset",
    "tell me about the data", "what is in the data", "what's in the data",
    "detailed summary", "give me a summary",
}

MAX_RESULT_ROWS = 20
MAX_SQL_RETRIES = 5


@dataclass
class SearchResult:
    """Unified return type for all search paths (SQL + PDF)."""
    reply: str
    vega_spec: dict | None = field(default=None)
    insights: list = field(default_factory=list)


def _is_summary_query(query: str) -> bool:
    q = query.lower().strip()
    return any(kw in q for kw in _SUMMARY_KEYWORDS)


def set_dataset_summary(summary: str) -> None:
    global _dataset_summary
    _dataset_summary = summary


# ── PDF PageIndex helpers (VectorLess BM25) ──────────────────────────────────

def set_mode(mode: str) -> None:
    global _current_mode
    _current_mode = mode


def load_pdf_index(pages: list[dict]) -> None:
    global _pdf_pages, _pdf_bm25
    _pdf_pages = pages
    tokenized = [page["text"].lower().split() for page in pages]
    _pdf_bm25 = BM25Okapi(tokenized)
    print(f"[pdf-index] built BM25 index over {len(pages)} pages")


def _store_pdf_conversation(query: str, page_refs: str, answer: str) -> None:
    _recent_pdf_turns.append((query, page_refs, answer))


def _format_recent_pdf_turns() -> str:
    if not _recent_pdf_turns:
        return ""
    lines = []
    for i, (q, p, a) in enumerate(_recent_pdf_turns):
        label = "MOST RECENT" if i == len(_recent_pdf_turns) - 1 else f"Turn {i + 1}"
        lines.append(f"[{label}]\nQ: {q}\nPages: {p}\nA: {a}")
    return "\n\n".join(lines)


def _pdf_search(query: str, top_k: int = 3) -> SearchResult:
    if _pdf_bm25 is None or not _pdf_pages:
        return SearchResult(reply="No PDF loaded. Please upload a PDF file first.")

    # BM25 scoring — no vectors, pure term-frequency / inverse-document-frequency
    tokens = query.lower().split()
    scores = _pdf_bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    context_parts = []
    matched_page_nums = []
    for idx in top_indices:
        if scores[idx] > 0:
            page = _pdf_pages[idx]
            context_parts.append(f"[Page {page['page_num']}]\n{page['text']}")
            matched_page_nums.append(str(page["page_num"]))

    if not context_parts:
        return SearchResult(reply="I couldn't find any relevant content in the PDF for your question.")

    context = "\n\n---\n\n".join(context_parts)

    # Inject recent conversation turns for follow-up awareness
    recent_str = _format_recent_pdf_turns()
    history_block = (
        f"Recent conversation history (oldest first; [MOST RECENT] is the direct predecessor):\n"
        f"{recent_str}\n\n"
        if recent_str else ""
    )

    from config import PDF_ANSWER_PROMPT
    answer_prompt = PDF_ANSWER_PROMPT.format(
        context=context,
        history=history_block,
        query=query,
    )
    answer = chat(answer_prompt)

    page_refs = ", ".join(f"p.{p}" for p in matched_page_nums)
    _store_pdf_conversation(query, page_refs, answer)
    print(f"[pdf-index] answered from pages: {page_refs}")

    return SearchResult(reply=answer)


# ── Shared LLM helper ────────────────────────────────────────────────────────


def chat(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt)
    ]
    response = groq_client.chat.completions.create(
        model=groq_chat_model,
        messages=messages
    )
    usage = response.usage
    print(f"[tokens] input={usage.prompt_tokens}  output={usage.completion_tokens}  total={usage.total_tokens}")
    return response.choices[0].message.content


# ── Text-to-SQL helpers ───────────────────────────────────────────────────────

def load_into_duckdb(df: pd.DataFrame) -> None:
    global _duck_conn
    _duck_conn = duckdb.connect()
    _duck_conn.register("data", df)


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]   # drop opening fence line
        cleaned = cleaned.rsplit("```", 1)[0]   # drop closing fence
    return cleaned.strip()


def _generate_chart_and_insights(df: pd.DataFrame, query: str) -> tuple[dict | None, list[str]]:
    """Single LLM call that returns both the Vega-Lite v5 spec and analytical insights.

    Returns (vega_spec_dict_or_None, insights_list).
    """
    if df is None or df.empty:
        return None, []

    schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    data_json = df.to_json(orient="records", date_format="iso")

    from config import CHART_AND_INSIGHTS_PROMPT
    prompt = CHART_AND_INSIGHTS_PROMPT.format(
        query=query,
        schema=schema_str,
        data_json=data_json,
    )

    raw = _strip_fences(chat(prompt)).strip()

    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None, []

        spec = payload.get("vega_spec")
        if not isinstance(spec, dict):
            spec = None

        insights = payload.get("insights", [])
        if not isinstance(insights, list):
            insights = []
        insights = [str(i) for i in insights]

        if spec:
            print(f"[chart] Vega-Lite spec generated  |  {len(insights)} insight(s)")
        return spec, insights

    except json.JSONDecodeError as exc:
        print(f"[chart] invalid JSON from LLM — skipping chart+insights: {exc}")
        return None, []


# ── Conversation memory helpers ──────────────────────────────────────────────

def _embed_for_memory(text: str) -> list[float] | None:
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"[memory] Ollama unavailable, skipping embedding: {e}")
        return None


def _store_conversation(query: str, sql: str, answer: str) -> None:
    # Sequential buffer — always reliable for immediate follow-ups
    _recent_turns.append((query, sql, answer))

    # Vector store — used as a secondary fallback for non-adjacent references
    vector = _embed_for_memory(query)
    if vector is None:
        return
    # Include a timestamp suffix so repeated identical queries don't overwrite
    import time
    doc_id = f"conv_{abs(hash(query))}_{int(time.time())}"
    _conv_collection.upsert(
        ids=[doc_id],
        embeddings=[vector],
        documents=[f"Q: {query}\nSQL: {sql}\nA: {answer}"],
    )
    print(f"[memory] stored conversation for: {query[:60]}")


def _format_recent_turns() -> str:
    if not _recent_turns:
        return ""
    lines = []
    for i, (q, s, a) in enumerate(_recent_turns):
        label = "MOST RECENT" if i == len(_recent_turns) - 1 else f"Turn {i + 1}"
        lines.append(f"[{label}]\nQ: {q}\nSQL: {s}\nA: {a}")
    return "\n\n".join(lines)


def _retrieve_similar_conversations(query: str, top_k: int = 3) -> str:
    vector = _embed_for_memory(query)
    if vector is None:
        return ""
    try:
        # Cap n_results to actual collection size to avoid ChromaDB crash
        count = _conv_collection.count()
        if count == 0:
            return ""
        results = _conv_collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
        )
        docs = results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"[memory] retrieval failed: {e}")
        return ""
    if not docs:
        return ""
    print(f"[memory] retrieved {len(docs)} relevant past interaction(s)")
    return "\n\n".join(docs)


def _text_to_sql_search(query: str) -> SearchResult:
    global _duck_conn

    if _duck_conn is None:
        return SearchResult(reply="No data loaded. Please upload a file first.")

    # Short-circuit: broad summary queries use the pre-generated dataset summary
    if _is_summary_query(query) and _dataset_summary:
        return SearchResult(reply=_dataset_summary)

    # Step 1 – extract schema from DuckDB via DESCRIBE
    schema_rows = _duck_conn.execute("DESCRIBE SELECT * FROM data").fetchall()
    schema_str = "\n".join(f"  {col}" for col, *_ in schema_rows)

    # Step 2 – build context block: sequential buffer first, vector fallback second
    recent_str = _format_recent_turns()
    recent_block = (
        f"Recent conversation history (oldest first; [MOST RECENT] is the direct predecessor):\n"
        f"{recent_str}\n\n"
        if recent_str else ""
    )

    past_context = _retrieve_similar_conversations(query)
    vector_block = (
        f"Additional semantically similar past interactions (lower priority than recent history above):\n{past_context}\n\n"
        if past_context else ""
    )

    memory_block = (
        (recent_block + vector_block)
        +
        "CRITICAL RULE FOR FOLLOW-UPS — When the question uses pronouns like 'them', 'those', "
        "'that room', 'those guests', etc., you MUST resolve them by wrapping the SQL from the "
        "[MOST RECENT] turn above as a subquery or CTE. "
        "NEVER hardcode room numbers, names, or any specific values copied from a text answer. "
        "Always compose the answer using SQL subqueries so the scope is exactly correct.\n\n"
        if recent_block or vector_block else ""
    )

    sql_prompt = (
        f"You have a DuckDB table called 'data' with this schema:\n{schema_str}\n\n"
        f"{memory_block}"
        f"Write a DuckDB SQL query to answer: {query}\n\n"
        "Rules:\n"
        "1. Use double quotes around all column names.\n"
        "2. SELECT all columns needed to fully answer the question — if filtering by a column, also include it in the SELECT clause to ensure the result is self-explanatory and complete.\n"
        "3. When a follow-up question refers to a subset from a previous interaction (using pronouns like 'those', 'that', 'them'), wrap the [MOST RECENT] SQL as a subquery and preserve its LIMIT and ORDER BY clauses exactly — do not expand the scope or add additional filtering.\n"
        "4. When the question refers to a single entity (indicated by singular articles like 'that' + noun), the subquery identifying it must use LIMIT 1 to ensure only one result is returned.\n"
        "Return only the raw SQL — no explanation, no markdown, no backticks."
    )
    sql = _strip_fences(chat(sql_prompt))

    # Agentic self-correction loop — retry up to MAX_SQL_RETRIES times
    result_df = None
    last_error = None
    for attempt in range(1, MAX_SQL_RETRIES + 1):
        try:
            result_df = _duck_conn.execute(sql).df()
            print(f"[sql-agent] query succeeded on attempt {attempt}")
            break
        except Exception as err:
            last_error = err
            print(f"[sql-agent] attempt {attempt}/{MAX_SQL_RETRIES} failed: {err}")
            if attempt < MAX_SQL_RETRIES:
                heal_prompt = (
                    f"The following DuckDB SQL query:\n{sql}\n\n"
                    f"Failed with this error:\n{err}\n\n"
                    f"Original question: {query}\n"
                    f"Table schema:\n{schema_str}\n\n"
                    "Diagnose the error, fix the query, and return only the corrected SQL — "
                    "no explanation, no markdown, no backticks."
                )
                sql = _strip_fences(chat(heal_prompt))

    if result_df is None:
        return SearchResult(
            reply=(
                f"I was unable to answer your question after {MAX_SQL_RETRIES} attempts. "
                f"Last error: {last_error}"
            )
        )

    # Truncate before both summary and chart generation
    total_rows = len(result_df)
    if total_rows > MAX_RESULT_ROWS:
        result_df = result_df.head(MAX_RESULT_ROWS)
        print(f"[token guard] result truncated to {MAX_RESULT_ROWS} of {total_rows} rows")

    # ── Natural-language summary ─────────────────────────────────────────────
    result_csv = result_df.to_csv(index=False)
    summary_prompt = (
        f"The following SQL was executed:\n{sql}\n\n"
        f"Query result:\n{result_csv}\n\n"
        f"Answer concisely in natural language: {query}\n\n"
        "IMPORTANT RULES:\n"
        "- The SQL and its result are already correct — trust them completely.\n"
        "- Never say you need more data — everything needed is in the result above.\n"
        "- Never summarize or truncate — return all rows from the result.\n"
        "- If the user asks to list all, list all — no exceptions."
    )
    summary = chat(summary_prompt)

    # ── Vega-Lite spec + analytical insights (single LLM call) ──────────────
    vega_spec, insights = _generate_chart_and_insights(result_df, query)

    # Store Q+SQL+A triplet in vector memory for future context retrieval
    _store_conversation(query, sql, summary)

    return SearchResult(reply=summary, vega_spec=vega_spec, insights=insights)


# ── Vector / embedding functions (preserved for future use) ──────────────────

def embed(text: str) -> list[float]:
    if USE_VECTOR_SEARCH:   # Preserved for future use
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    return []


def store(texts: list[str], doc_id: list[str]) -> None:
    if USE_VECTOR_SEARCH:   # Preserved for future use
        vectors = [embed(text) for text in texts]
        collection.add(ids=doc_id, embeddings=vectors, documents=texts)


def store_one(text: str, doc_id: str) -> None:
    if USE_VECTOR_SEARCH:   # Preserved for future use
        store([text], [doc_id])


def search(query: str, top_k: int = 3) -> SearchResult:
    # PDF mode: VectorLess BM25 PageIndex retrieval
    if _current_mode == "pdf":
        return _pdf_search(query, top_k)

    if USE_VECTOR_SEARCH:   # Preserved for future use
        query_vector = embed(query)
        results = collection.query(query_embeddings=[query_vector], n_results=top_k)
        return SearchResult(reply="\n\n".join(results["documents"][0]))

    # Text-to-SQL path (Excel / CSV)
    return _text_to_sql_search(query)


def delete() -> None:
    global _duck_conn, _pdf_pages, _pdf_bm25, _current_mode
    # Clear all conversation buffers so prior context doesn't bleed into a new session
    _recent_turns.clear()
    _recent_pdf_turns.clear()
    # Reset PDF state
    _pdf_pages = None
    _pdf_bm25 = None
    _current_mode = "excel"
    if USE_VECTOR_SEARCH:   # Preserved for future use
        global collection
        _chroma_client.delete_collection("my_collection")
        collection = _chroma_client.get_or_create_collection("my_collection")
    else:
        # Reset the in-memory DuckDB so stale data from a previous upload is gone
        _duck_conn = None