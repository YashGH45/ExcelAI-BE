from session import Session
from intent import QueryIntent
from llm import get_response_from_llm, strip_fences
from metadata import retrieve_relevant_metadata
from memory import store_conversation, format_recent_turns, retrieve_similar_conversations
from charts import generate_chart_and_insights
from dictionary import _SUMMARY_KEYWORDS
from dto_models import SearchResult

MAX_RESULT_ROWS = 20
MAX_SQL_RETRIES = 5

_ANALYTICAL_INTENTS = {
    QueryIntent.ANSWER_WITH_CHART,
    QueryIntent.ANSWER_WITH_INSIGHTS,
    QueryIntent.ANSWER_WITH_CHART_AND_INSIGHTS,
}


def _is_summary_query(query: str) -> bool:
    q = query.lower().strip()
    return any(kw in q for kw in _SUMMARY_KEYWORDS)


def _build_memory_block(query: str, session: Session) -> str:
    recent_str = format_recent_turns(session)
    summary_block = (
        f"Summary of earlier conversation:\n{session.conversation_summary}\n\n"
        if session.conversation_summary else ""
    )
    recent_block = (
        f"Recent conversation history (oldest first; [MOST RECENT] is the direct predecessor):\n"
        f"{recent_str}\n\n"
        if recent_str else ""
    )
    past_context = retrieve_similar_conversations(query, session)
    vector_block = (
        f"Additional semantically similar past interactions (lower priority than recent history above):\n{past_context}\n\n"
        if past_context else ""
    )

    if not (summary_block or recent_block or vector_block):
        return ""

    return (
        summary_block + recent_block + vector_block +
        "CRITICAL RULE FOR FOLLOW-UPS — When the question uses pronouns like 'them', 'those', "
        "'that room', 'those guests', etc., you MUST resolve them by wrapping the SQL from the "
        "[MOST RECENT] turn above as a subquery or CTE. "
        "NEVER hardcode room numbers, names, or any specific values copied from a text answer. "
        "Always compose the answer using SQL subqueries so the scope is exactly correct.\n\n"
    )


def _build_metadata_section(query: str, session: Session, intent: QueryIntent) -> str:
    if intent not in _ANALYTICAL_INTENTS:
        print("[metadata] skipping chunk retrieval for non-analytical intent")
        return ""
    metadata_block = retrieve_relevant_metadata(query, session)
    if not metadata_block:
        return ""
    return (
        f"Rich column metadata (use descriptions and aliases to map user terms to correct column names):\n"
        f"{metadata_block}\n\n"
    )


def _generate_sql(query: str, schema_str: str, metadata_section: str, memory_block: str) -> str:
    prompt = (
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
    return strip_fences(get_response_from_llm(prompt))


def _execute_with_retry(sql: str, query: str, schema_str: str, session: Session):
    """Returns (result_df, final_sql, last_error). result_df is None on total failure."""
    last_error = None
    for attempt in range(1, MAX_SQL_RETRIES + 1):
        try:
            result_df = session.duck_conn.execute(sql).df()
            print(f"[sql-agent] query succeeded on attempt {attempt}")
            return result_df, sql, None
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
                sql = strip_fences(get_response_from_llm(heal_prompt))
    return None, sql, last_error


def _apply_fuzzy_fallback(result_df, sql: str, query: str, schema_str: str, session: Session):
    if not result_df.empty:
        return result_df, sql

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
    fuzzy_sql = strip_fences(get_response_from_llm(fuzzy_prompt))
    try:
        fuzzy_df = session.duck_conn.execute(fuzzy_sql).df()
        if not fuzzy_df.empty:
            print(f"[sql-agent] fuzzy retry found {len(fuzzy_df)} row(s)")
            return fuzzy_df, fuzzy_sql
        print("[sql-agent] fuzzy retry also returned 0 rows")
    except Exception as fuzzy_err:
        print(f"[sql-agent] fuzzy retry failed: {fuzzy_err}")
    return result_df, sql


def _summarize_result(result_df, sql: str, query: str) -> str:
    result_csv = result_df.to_csv(index=False)
    prompt = (
        f"The following SQL was executed:\n{sql}\n\n"
        f"Query result:\n{result_csv}\n\n"
        f"Answer concisely in natural language: {query}\n\n"
        "IMPORTANT RULES:\n"
        "- The SQL and its result are already correct — trust them completely.\n"
        "- Never say you need more data — everything needed is in the result above.\n"
        "- Never summarize or truncate — return all rows from the result.\n"
        "- If the user asks to list all, list all — no exceptions."
    )
    return get_response_from_llm(prompt)


def text_to_sql_search(query: str, session: Session, intent: QueryIntent) -> SearchResult:
    if session.duck_conn is None:
        return SearchResult(reply="No data loaded. Please upload a file first.")

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

    memory_block = _build_memory_block(query, session)
    metadata_section = _build_metadata_section(query, session, intent)
    sql = _generate_sql(query, schema_str, metadata_section, memory_block)

    result_df, sql, last_error = _execute_with_retry(sql, query, schema_str, session)
    if result_df is None:
        return SearchResult(
            reply=(
                f"I was unable to answer your question after {MAX_SQL_RETRIES} attempts. "
                f"Last error: {last_error}"
            )
        )

    result_df, sql = _apply_fuzzy_fallback(result_df, sql, query, schema_str, session)

    total_rows = len(result_df)
    if total_rows > MAX_RESULT_ROWS:
        result_df = result_df.head(MAX_RESULT_ROWS)
        print(f"[token guard] result truncated to {MAX_RESULT_ROWS} of {total_rows} rows")

    summary = _summarize_result(result_df, sql, query)

    wants_chart = intent in (QueryIntent.ANSWER_WITH_CHART, QueryIntent.ANSWER_WITH_CHART_AND_INSIGHTS)
    wants_insights = intent in (QueryIntent.ANSWER_WITH_INSIGHTS, QueryIntent.ANSWER_WITH_CHART_AND_INSIGHTS)

    if (wants_chart or wants_insights) and len(result_df) <= 1:
        wants_chart = False
        wants_insights = False
        print("[intent] downgraded to answer-only — result has ≤ 1 row")

    if wants_chart or wants_insights:
        vega_spec, insights = generate_chart_and_insights(
            result_df, query,
            want_chart=wants_chart,
            want_insights=wants_insights,
        )
    else:
        vega_spec, insights = None, []
        print(f"[intent] skipping chart+insights for intent={intent.value}")

    store_conversation(query, sql, summary, session)

    return SearchResult(reply=summary, vega_spec=vega_spec, insights=insights)
