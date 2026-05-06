import os
import json
import time
import duckdb
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from dotenv import load_dotenv
from groq import Groq
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
import ollama
from rank_bm25 import BM25Okapi

from session import Session

load_dotenv()

USE_VECTOR_SEARCH = False

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
groq_chat_model = os.environ["GROQ_CHAT_MODEL"]

MAX_RESULT_ROWS = 20
MAX_SQL_RETRIES = 5

_SUMMARY_KEYWORDS = {
    "summary", "summarize", "summarise", "summarization",
    "overview", "describe the data", "describe the dataset",
    "tell me about the data", "what is in the data", "what's in the data",
    "detailed summary", "give me a summary",
}


# ── Query intent classification ──────────────────────────────────────────────

class QueryIntent(Enum):
    ANSWER_ONLY = "answer_only"
    ANSWER_WITH_CHART = "answer_with_chart"
    ANSWER_WITH_INSIGHTS = "answer_with_insights"
    ANSWER_WITH_CHART_AND_INSIGHTS = "answer_with_chart_and_insights"
    WRITE = "write"
    CLEANSE = "cleanse"


_CLEANSE_KEYWORDS = (
    "cleanse", "clean the data", "clean the file", "clean up the data",
    "clean up the file", "fix the data", "fix dirty", "fix missing values",
    "fix errors in the data", "correct the data", "data cleansing",
    "data cleaning", "sanitize the data", "sanitise the data",
    "repair the data", "remove dirty rows", "handle missing",
    "impute missing", "fix null", "fix nulls",
)

_WRITE_PHRASES = (
    "update the", "update this", "modify the", "modify this",
    "change the value", "change this value", "set the value", "set this to",
    "delete the row", "delete this row", "remove the row", "remove this row",
    "insert a row", "add a row", "add row", "edit the", "edit this",
    "overwrite", "replace the value", "write to the",
)

_CHART_KEYWORDS = (
    "plot", "graph", "chart", "visualize", "visualise", "visualization",
    "visualisation", "draw", "histogram", "bar chart", "pie chart",
    "line chart", "scatter plot", "show me a chart", "show a chart",
)

_INSIGHTS_KEYWORDS = (
    "insight", "insights", "analyze", "analyse", "analysis", "trend",
    "trends", "pattern", "patterns", "notable", "interesting", "highlight",
    "observation", "takeaway", "stands out", "key finding",
)

_AGGREGATION_KEYWORDS = (
    "total", "average", "avg", "sum", "maximum", "minimum",
    "distribution", "breakdown", "compare", "comparison", "ranking", "rank",
    "how many", "how much", "percentage", "percent",
    "by month", "by week", "by year", "by quarter", "by day",
    "per month", "per year", "per day", "per quarter",
    "over time", "year over year", "month over month",
    "most popular", "least popular", "highest", "lowest", "most", "least",
    "top 5", "top 10", "bottom 5", "bottom 10",
)

_NON_AGGREGATION_BY_PHRASES = (
    "sort by", "order by", "sorted by", "ordered by", "filter by", "filtered by",
    "divided by", "caused by",
)


def classify_intent(query: str) -> QueryIntent:
    q = query.lower().strip()

    if any(kw in q for kw in _CLEANSE_KEYWORDS):
        return QueryIntent.CLEANSE

    if any(phrase in q for phrase in _WRITE_PHRASES):
        return QueryIntent.WRITE

    has_chart = any(kw in q for kw in _CHART_KEYWORDS)
    has_insights = any(kw in q for kw in _INSIGHTS_KEYWORDS)

    if has_chart and has_insights:
        return QueryIntent.ANSWER_WITH_CHART_AND_INSIGHTS
    if has_chart:
        return QueryIntent.ANSWER_WITH_CHART
    if has_insights:
        return QueryIntent.ANSWER_WITH_INSIGHTS

    has_aggregation = any(kw in q for kw in _AGGREGATION_KEYWORDS) or (
        " by " in q
        and not any(phrase in q for phrase in _NON_AGGREGATION_BY_PHRASES)
    )
    if has_aggregation:
        return QueryIntent.ANSWER_WITH_CHART_AND_INSIGHTS

    return QueryIntent.ANSWER_ONLY


@dataclass
class SearchResult:
    """Unified return type for all search paths (SQL + PDF)."""
    reply: str
    vega_spec: dict | None = field(default=None)
    insights: list = field(default_factory=list)
    download_url: str | None = field(default=None)


def _is_summary_query(query: str) -> bool:
    q = query.lower().strip()
    return any(kw in q for kw in _SUMMARY_KEYWORDS)


# ── Shared LLM helper ────────────────────────────────────────────────────────

def chat(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]
    response = groq_client.chat.completions.create(
        model=groq_chat_model,
        messages=messages,
    )
    usage = response.usage
    print(f"[tokens] input={usage.prompt_tokens}  output={usage.completion_tokens}  total={usage.total_tokens}")
    return response.choices[0].message.content


# ── DuckDB ───────────────────────────────────────────────────────────────────

def load_into_duckdb(df: pd.DataFrame, session: Session) -> None:
    session.duck_conn = duckdb.connect()
    session.duck_conn.register("data", df)


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


# ── Metadata helpers ─────────────────────────────────────────────────────────

def _col_to_doc(col: dict) -> str:
    parts = [
        f"Column: {col.get('column', '')}",
        f"Type: {col.get('type', '')}",
        f"Description: {col.get('description', '')}",
    ]
    examples = col.get("example_values", [])
    if examples:
        parts.append(f"Example values: {', '.join(str(v) for v in examples)}")
    aliases = col.get("aliases", [])
    if aliases:
        parts.append(f"User may refer to this column as: {', '.join(aliases)}")
    related = col.get("related_columns", [])
    if related:
        parts.append(f"Related columns: {', '.join(related)}")
    return "\n".join(parts)


def _table_to_doc(table: dict) -> str:
    parts = []
    if table.get("description"):
        parts.append(f"Table description: {table['description']}")
    if table.get("grain"):
        parts.append(f"Grain: {table['grain']}")
    if table.get("time_range"):
        parts.append(f"Time range: {table['time_range']}")
    caveats = table.get("caveats", [])
    if caveats:
        parts.append(f"Caveats: {'; '.join(caveats)}")
    return "\n".join(parts)


def store_metadata_in_vectordb(metadata: dict, session: Session) -> None:
    session.rich_metadata = metadata

    try:
        existing_ids = session.metadata_collection.get()["ids"]
        if existing_ids:
            session.metadata_collection.delete(ids=existing_ids)
    except Exception as e:
        print(f"[metadata] failed to clear existing documents: {e}")

    documents: list[str] = []
    ids: list[str] = []

    table = metadata.get("table", {})
    if table:
        doc = _table_to_doc(table)
        if doc:
            documents.append(doc)
            ids.append("table_meta")

    for col in metadata.get("columns", []):
        name = col.get("column", "")
        if not name:
            continue
        documents.append(_col_to_doc(col))
        ids.append(f"col_{name}")

    if documents:
        session.metadata_collection.add(documents=documents, ids=ids)
        print(f"[metadata] stored {len(documents)} document(s) in vector DB ({len(documents) - 1} columns + 1 table)")


def _retrieve_relevant_metadata(query: str, session: Session, top_k: int = 6) -> str:
    try:
        count = session.metadata_collection.count()
        if count == 0:
            return ""
        results = session.metadata_collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
        )
        docs: list[str] = results["documents"][0] if results["documents"] else []
        if not docs:
            return ""
        print(f"[metadata] retrieved {len(docs)} relevant metadata chunk(s)")
        return "\n\n".join(docs)
    except Exception as e:
        print(f"[metadata] retrieval failed, falling back to full metadata dump: {e}")
        return _format_full_metadata(session)


def _format_full_metadata(session: Session) -> str:
    if not session.rich_metadata:
        return ""
    parts: list[str] = []
    table = session.rich_metadata.get("table", {})
    if table:
        parts.append(_table_to_doc(table))
    for col in session.rich_metadata.get("columns", []):
        parts.append(_col_to_doc(col))
    return "\n\n".join(parts)


# ── PDF helpers ───────────────────────────────────────────────────────────────

def load_pdf_index(pages: list[dict], session: Session) -> None:
    session.pdf_pages = pages
    tokenized = [page["text"].lower().split() for page in pages]
    session.pdf_bm25 = BM25Okapi(tokenized)
    print(f"[pdf-index] built BM25 index over {len(pages)} pages")


def _store_pdf_conversation(query: str, page_refs: str, answer: str, session: Session) -> None:
    if len(session.recent_pdf_turns) == session.recent_pdf_turns.maxlen:
        oldest_q, oldest_p, oldest_a = session.recent_pdf_turns[0]
        compress_prompt = (
            (f"Existing summary:\n{session.pdf_conversation_summary}\n\n" if session.pdf_conversation_summary else "") +
            f"New turn to fold in:\nQ: {oldest_q}\nA: {oldest_a}\n\n"
            "Update the summary in 2-3 sentences preserving key facts and page references. "
            "Return only the updated summary text, nothing else."
        )
        session.pdf_conversation_summary = chat(compress_prompt)
        print(f"[memory] pdf summary updated ({len(session.pdf_conversation_summary)} chars)")
    session.recent_pdf_turns.append((query, page_refs, answer))


def _format_recent_pdf_turns(session: Session) -> str:
    if not session.recent_pdf_turns:
        return ""
    lines = []
    for i, (q, p, a) in enumerate(session.recent_pdf_turns):
        label = "MOST RECENT" if i == len(session.recent_pdf_turns) - 1 else f"Turn {i + 1}"
        lines.append(f"[{label}]\nQ: {q}\nPages: {p}\nA: {a}")
    return "\n\n".join(lines)


def _pdf_search(query: str, session: Session, top_k: int = 3) -> SearchResult:
    if session.pdf_bm25 is None or not session.pdf_pages:
        return SearchResult(reply="No PDF loaded. Please upload a PDF file first.")

    tokens = query.lower().split()
    scores = session.pdf_bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    context_parts = []
    matched_page_nums = []
    for idx in top_indices:
        if scores[idx] > 0:
            page = session.pdf_pages[idx]
            context_parts.append(f"[Page {page['page_num']}]\n{page['text']}")
            matched_page_nums.append(str(page["page_num"]))

    if not context_parts:
        return SearchResult(reply="I couldn't find any relevant content in the PDF for your question.")

    context = "\n\n---\n\n".join(context_parts)

    recent_str = _format_recent_pdf_turns(session)
    summary_block = (
        f"Summary of earlier conversation:\n{session.pdf_conversation_summary}\n\n"
        if session.pdf_conversation_summary else ""
    )
    recent_block = (
        f"Recent conversation history (oldest first; [MOST RECENT] is the direct predecessor):\n"
        f"{recent_str}\n\n"
        if recent_str else ""
    )
    history_block = summary_block + recent_block

    from config import PDF_ANSWER_PROMPT
    answer_prompt = PDF_ANSWER_PROMPT.format(
        context=context,
        history=history_block,
        query=query,
    )
    answer = chat(answer_prompt)

    page_refs = ", ".join(f"p.{p}" for p in matched_page_nums)
    _store_pdf_conversation(query, page_refs, answer, session)
    print(f"[pdf-index] answered from pages: {page_refs}")

    return SearchResult(reply=answer)


# ── Data cleansing ────────────────────────────────────────────────────────────

_NULL_LIKE_STRINGS = frozenset({
    "n/a", "na", "null", "none", "nan", "#n/a", "#na", "-", "--",
    "?", "unknown", "missing", "undefined", "nil", "void", "",
})

MAX_DIRTY_BATCH = 50


def _identify_dirty_rows(df: pd.DataFrame) -> pd.Index:
    dirty_mask = pd.Series(False, index=df.index)

    dirty_mask |= df.isnull().any(axis=1)

    for col in df.select_dtypes(include="object").columns:
        col_lower = df[col].astype(str).str.lower().str.strip()
        dirty_mask |= col_lower.isin(_NULL_LIKE_STRINGS)

    for col in df.select_dtypes(include="object").columns:
        numeric_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
        if numeric_count / len(df) >= 0.7:
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            not_null_like = ~df[col].astype(str).str.lower().str.strip().isin(_NULL_LIKE_STRINGS)
            dirty_mask |= non_numeric & not_null_like

    return df.index[dirty_mask]


def cleanse_data(session: Session) -> SearchResult:
    if session.duck_conn is None:
        return SearchResult(reply="No data loaded. Please upload a file first.")
    if not session.current_filepath:
        return SearchResult(reply="Source file path not found. Please re-upload the file.")

    df = session.duck_conn.execute("SELECT * FROM data").df()
    dirty_idx = _identify_dirty_rows(df)

    if len(dirty_idx) == 0:
        return SearchResult(reply="The dataset looks clean — no dirty rows were detected!")

    dirty_df = df.loc[dirty_idx]
    clean_df = df.drop(dirty_idx)

    reference_df = clean_df.dropna().head(5)
    if reference_df.empty:
        reference_df = df.head(5)

    schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    reference_json = reference_df.to_json(orient="records", date_format="iso")

    from config import CLEANSE_PROMPT

    corrected_rows: list[dict] = []
    total_batches = (len(dirty_df) + MAX_DIRTY_BATCH - 1) // MAX_DIRTY_BATCH

    for batch_num, start in enumerate(range(0, len(dirty_df), MAX_DIRTY_BATCH), 1):
        batch = dirty_df.iloc[start : start + MAX_DIRTY_BATCH].copy()
        batch.insert(0, "_row_index", batch.index.tolist())
        batch_json = batch.to_json(orient="records", date_format="iso")

        print(f"[cleanse] batch {batch_num}/{total_batches} — {len(batch)} dirty rows")

        prompt = CLEANSE_PROMPT.format(
            schema=schema_str,
            reference_rows=reference_json,
            dirty_rows=batch_json,
        )
        raw = _strip_fences(chat(prompt)).strip()

        try:
            result = json.loads(raw)
            if isinstance(result, list):
                corrected_rows.extend(result)
            else:
                print(f"[cleanse] batch {batch_num} — unexpected response structure, skipping")
        except json.JSONDecodeError as exc:
            print(f"[cleanse] batch {batch_num} — invalid JSON: {exc}")

    result_df = df.copy()
    applied = 0
    for row in corrected_rows:
        idx = row.pop("_row_index", None)
        if idx is None or idx not in result_df.index:
            continue
        for col, val in row.items():
            if col in result_df.columns:
                result_df.at[idx, col] = val
        applied += 1

    base, ext = os.path.splitext(session.current_filepath)
    cleansed_path = f"{base}_cleansed{ext}"
    cleansed_filename = os.path.basename(cleansed_path)

    if ext.lower() == ".csv":
        result_df.to_csv(cleansed_path, index=False)
    else:
        result_df.to_excel(cleansed_path, index=False)

    print(f"[cleanse] saved → {cleansed_path}")

    summary = (
        f"Data cleansing complete!\n\n"
        f"- **{len(dirty_idx)}** dirty rows detected out of **{len(df)}** total\n"
        f"- **{applied}** rows corrected by the AI\n"
        f"- Cleansed file: `{cleansed_filename}`\n\n"
        f"Use the download button below to get the cleansed file."
    )
    return SearchResult(
        reply=summary,
        download_url=f"/api/download/{cleansed_filename}",
    )


# ── Chart and insights ────────────────────────────────────────────────────────

def _generate_chart_and_insights(
    df: pd.DataFrame,
    query: str,
    want_chart: bool = True,
    want_insights: bool = True,
) -> tuple[dict | None, list[str]]:
    if df is None or df.empty or (not want_chart and not want_insights):
        return None, []

    schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    data_json = df.to_json(orient="records", date_format="iso")

    from config import CHART_AND_INSIGHTS_PROMPT, CHART_ONLY_PROMPT, INSIGHTS_ONLY_PROMPT

    if want_chart and want_insights:
        prompt = CHART_AND_INSIGHTS_PROMPT.format(query=query, schema=schema_str, data_json=data_json)
    elif want_chart:
        prompt = CHART_ONLY_PROMPT.format(query=query, schema=schema_str, data_json=data_json)
    else:
        prompt = INSIGHTS_ONLY_PROMPT.format(query=query, schema=schema_str, data_json=data_json)

    raw = _strip_fences(chat(prompt)).strip()

    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None, []

        spec = payload.get("vega_spec") if want_chart else None
        if not isinstance(spec, dict):
            spec = None

        insights = payload.get("insights", []) if want_insights else []
        if not isinstance(insights, list):
            insights = []
        insights = [str(i) for i in insights]

        if spec:
            print(f"[chart] Vega-Lite spec generated  |  {len(insights)} insight(s)")
        elif want_insights:
            print(f"[chart] {len(insights)} insight(s) generated (no chart)")
        return spec, insights

    except json.JSONDecodeError as exc:
        print(f"[chart] invalid JSON from LLM — skipping: {exc}")
        return None, []


# ── Conversation memory ───────────────────────────────────────────────────────

def _embed_for_memory(text: str) -> list[float] | None:
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"[memory] Ollama unavailable, skipping embedding: {e}")
        return None


def _store_conversation(query: str, sql: str, answer: str, session: Session) -> None:
    if len(session.recent_turns) == session.recent_turns.maxlen:
        oldest_q, oldest_sql, oldest_a = session.recent_turns[0]
        compress_prompt = (
            (f"Existing summary:\n{session.conversation_summary}\n\n" if session.conversation_summary else "") +
            f"New turn to fold in:\nQ: {oldest_q}\nA: {oldest_a}\n\n"
            "Update the summary in 2-3 sentences preserving key facts, column names, "
            "filter values, and numeric results. Return only the updated summary text, nothing else."
        )
        session.conversation_summary = chat(compress_prompt)
        print(f"[memory] summary updated ({len(session.conversation_summary)} chars)")

    session.recent_turns.append((query, sql, answer))

    vector = _embed_for_memory(query)
    if vector is None:
        return
    doc_id = f"conv_{abs(hash(query))}_{int(time.time())}"
    session.conv_collection.upsert(
        ids=[doc_id],
        embeddings=[vector],
        documents=[f"Q: {query}\nSQL: {sql}\nA: {answer}"],
    )
    print(f"[memory] stored conversation for: {query[:60]}")


def _format_recent_turns(session: Session) -> str:
    if not session.recent_turns:
        return ""
    lines = []
    for i, (q, s, a) in enumerate(session.recent_turns):
        label = "MOST RECENT" if i == len(session.recent_turns) - 1 else f"Turn {i + 1}"
        lines.append(f"[{label}]\nQ: {q}\nSQL: {s}\nA: {a}")
    return "\n\n".join(lines)


def _retrieve_similar_conversations(query: str, session: Session, top_k: int = 3) -> str:
    vector = _embed_for_memory(query)
    if vector is None:
        return ""
    try:
        count = session.conv_collection.count()
        if count == 0:
            return ""
        results = session.conv_collection.query(
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


# ── Text-to-SQL ───────────────────────────────────────────────────────────────

def _text_to_sql_search(query: str, session: Session) -> SearchResult:
    if session.duck_conn is None:
        return SearchResult(reply="No data loaded. Please upload a file first.")

    intent = classify_intent(query)
    print(f"[intent] {intent.value}")

    if intent == QueryIntent.WRITE:
        return SearchResult(
            reply=(
                "Write operations on uploaded data are not yet supported. "
                "This feature is coming soon!"
            )
        )

    if _is_summary_query(query) and session.dataset_summary:
        return SearchResult(reply=session.dataset_summary)

    schema_rows = session.duck_conn.execute("DESCRIBE SELECT * FROM data").fetchall()
    schema_str = "\n".join(f"  {col}" for col, *_ in schema_rows)

    recent_str = _format_recent_turns(session)
    summary_block = (
        f"Summary of earlier conversation:\n{session.conversation_summary}\n\n"
        if session.conversation_summary else ""
    )
    recent_block = (
        f"Recent conversation history (oldest first; [MOST RECENT] is the direct predecessor):\n"
        f"{recent_str}\n\n"
        if recent_str else ""
    )

    past_context = _retrieve_similar_conversations(query, session)
    vector_block = (
        f"Additional semantically similar past interactions (lower priority than recent history above):\n{past_context}\n\n"
        if past_context else ""
    )

    has_context = bool(summary_block or recent_block or vector_block)
    memory_block = (
        (summary_block + recent_block + vector_block)
        +
        "CRITICAL RULE FOR FOLLOW-UPS — When the question uses pronouns like 'them', 'those', "
        "'that room', 'those guests', etc., you MUST resolve them by wrapping the SQL from the "
        "[MOST RECENT] turn above as a subquery or CTE. "
        "NEVER hardcode room numbers, names, or any specific values copied from a text answer. "
        "Always compose the answer using SQL subqueries so the scope is exactly correct.\n\n"
        if has_context else ""
    )

    metadata_block = _retrieve_relevant_metadata(query, session)
    metadata_section = (
        f"Rich column metadata (use descriptions and aliases to map user terms to correct column names):\n"
        f"{metadata_block}\n\n"
        if metadata_block else ""
    )

    sql_prompt = (
        f"You have a DuckDB table called 'data' with this schema:\n{schema_str}\n\n"
        f"{metadata_section}"
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

    result_df = None
    last_error = None
    for attempt in range(1, MAX_SQL_RETRIES + 1):
        try:
            result_df = session.duck_conn.execute(sql).df()
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

    if result_df.empty:
        print("[sql-agent] exact query returned 0 rows — retrying with fuzzy ILIKE match")
        fuzzy_prompt = (
            f"The following DuckDB SQL query:\n{sql}\n\n"
            f"Ran successfully but returned 0 rows for the question: {query}\n"
            f"Table schema:\n{schema_str}\n\n"
            "The string filter likely missed due to case differences or slight name variations. "
            "Rewrite the query replacing any exact string equality (=) filters on text/name columns "
            "with case-insensitive partial matches using ILIKE '%value%'. "
            "Keep all other parts of the query identical. "
            "Return only the corrected SQL — no explanation, no markdown, no backticks."
        )
        fuzzy_sql = _strip_fences(chat(fuzzy_prompt))
        try:
            fuzzy_df = session.duck_conn.execute(fuzzy_sql).df()
            if not fuzzy_df.empty:
                result_df = fuzzy_df
                sql = fuzzy_sql
                print(f"[sql-agent] fuzzy retry found {len(fuzzy_df)} row(s)")
            else:
                print("[sql-agent] fuzzy retry also returned 0 rows")
        except Exception as fuzzy_err:
            print(f"[sql-agent] fuzzy retry failed: {fuzzy_err}")

    if result_df is None:
        return SearchResult(reply="Unexpected error: result was lost after fuzzy retry.")

    total_rows = len(result_df)
    if total_rows > MAX_RESULT_ROWS:
        result_df = result_df.head(MAX_RESULT_ROWS)
        print(f"[token guard] result truncated to {MAX_RESULT_ROWS} of {total_rows} rows")

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

    wants_chart = intent in (QueryIntent.ANSWER_WITH_CHART, QueryIntent.ANSWER_WITH_CHART_AND_INSIGHTS)
    wants_insights = intent in (QueryIntent.ANSWER_WITH_INSIGHTS, QueryIntent.ANSWER_WITH_CHART_AND_INSIGHTS)

    if (wants_chart or wants_insights) and len(result_df) <= 1:
        wants_chart = False
        wants_insights = False
        print("[intent] downgraded to answer-only — result has ≤ 1 row")

    if wants_chart or wants_insights:
        vega_spec, insights = _generate_chart_and_insights(
            result_df, query,
            want_chart=wants_chart,
            want_insights=wants_insights,
        )
    else:
        vega_spec, insights = None, []
        print(f"[intent] skipping chart+insights for intent={intent.value}")

    _store_conversation(query, sql, summary, session)

    return SearchResult(reply=summary, vega_spec=vega_spec, insights=insights)


# ── Main entry point ──────────────────────────────────────────────────────────

def search(query: str, session: Session, top_k: int = 3) -> SearchResult:
    if session.current_mode == "pdf":
        return _pdf_search(query, session, top_k)

    intent = classify_intent(query)
    if intent == QueryIntent.CLEANSE:
        return cleanse_data(session)

    return _text_to_sql_search(query, session)


# ── Stub preserved for services.py backward compatibility ────────────────────

def store(texts: list[str], doc_id: list[str]) -> None:
    """No-op — vector search is disabled (USE_VECTOR_SEARCH = False)."""
    pass