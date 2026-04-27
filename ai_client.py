import os
import duckdb
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
import ollama
import chromadb

load_dotenv()

USE_VECTOR_SEARCH = False

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
groq_chat_model = os.environ["GROQ_CHAT_MODEL"]
groq_mini_chat_model = os.environ["GROQ_MINI_CHAT_MODEL"]
gemma_model = os.environ["GEMMA_MODEL"]

if USE_VECTOR_SEARCH:
    chroma_client = chromadb.PersistentClient(path="./vector_db")
    collection = chroma_client.get_or_create_collection(name="my_collection")

_duck_conn: duckdb.DuckDBPyConnection | None = None
_dataset_summary: str | None = None

_SUMMARY_KEYWORDS = {
    "summary", "summarize", "summarise", "summarization",
    "overview", "describe the data", "describe the dataset",
    "tell me about the data", "what is in the data", "what's in the data",
    "detailed summary", "give me a summary",
}

MAX_RESULT_ROWS = 20


def _is_summary_query(query: str) -> bool:
    q = query.lower().strip()
    return any(kw in q for kw in _SUMMARY_KEYWORDS)


def set_dataset_summary(summary: str) -> None:
    global _dataset_summary
    _dataset_summary = summary


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


def _text_to_sql_search(query: str) -> list[str]:
    global _duck_conn

    if _duck_conn is None:
        return ["No data loaded. Please upload a file first."]

    # Short-circuit: broad summary queries use the pre-generated dataset summary
    if _is_summary_query(query) and _dataset_summary:
        return [_dataset_summary]

    # Step 1 – extract schema from DuckDB via DESCRIBE
    schema_rows = _duck_conn.execute("DESCRIBE SELECT * FROM data").fetchall()
    schema_str = "\n".join(f"  {col}" for col, *_ in schema_rows)

    sql_prompt = (
        f"You have a DuckDB table called 'data' with this schema:\n{schema_str}\n\n"
        f"Write a DuckDB SQL query to answer: {query}\n\n"
        "using double quotes around all column names"
        "Return only the raw SQL — no explanation, no markdown, no backticks."
    )
    raw_sql = chat(sql_prompt)
    sql = _strip_fences(raw_sql)

    # execute, on failure feed the error back for one retry
    try:
        result_df = _duck_conn.execute(sql).df()
    except Exception as first_err:
        retry_prompt = (
            f"The SQL query failed with: {first_err}\n\n"
            f"Original question: {query}\n"
            f"Schema:\n{schema_str}\n\n"
            "Return only the corrected SQL — no markdown, no backticks."
        )
        sql = _strip_fences(chat(retry_prompt))
        result_df = _duck_conn.execute(sql).df()

    # summarise the result in natural language using the LLM
    total_rows = len(result_df)
    if total_rows > MAX_RESULT_ROWS:
        result_df = result_df.head(MAX_RESULT_ROWS)
        print(f"[token guard] result truncated to {MAX_RESULT_ROWS} of {total_rows} rows")
    result_csv = result_df.to_csv(index=False)
    summary_prompt = (
        f"Given this query result:\n{result_csv}\n\n"
        f"Answer concisely in natural language: {query}"
        """IMPORTANT RULES:
            - Never summarize or truncate query results
            - Never make decisions on behalf of the user about what is 'too much' data
            - Always return the exact data from the query result
            - If the user asks to list all, list all — no exceptions"""
    )
    summary = chat(summary_prompt)

    return [summary]


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


def search(query: str, top_k: int = 3) -> list[str]:
    if USE_VECTOR_SEARCH:   # Preserved for future use
        query_vector = embed(query)
        results = collection.query(query_embeddings=[query_vector], n_results=top_k)
        return results["documents"][0]

    # Text-to-SQL path
    return _text_to_sql_search(query)


def delete() -> None:
    global _duck_conn
    if USE_VECTOR_SEARCH:   # Preserved for future use
        global collection
        chroma_client.delete_collection("my_collection")
        collection = chroma_client.get_or_create_collection("my_collection")
    else:
        # Reset the in-memory DuckDB so stale data from a previous upload is gone
        _duck_conn = None