import json
import pandas as pd
from session import Session
from llm import get_response_from_llm, strip_fences


# ── Document formatters ───────────────────────────────────────────────────────

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


# ── Vector store operations ───────────────────────────────────────────────────

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


def retrieve_relevant_metadata(query: str, session: Session, top_k: int = 6) -> str:
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


# ── LLM-based generation ──────────────────────────────────────────────────────

def generate_rich_metadata(df: pd.DataFrame, answers: list[dict] | None = None) -> dict:

    from config import RICH_METADATA_PROMPT
    schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    sample_df = df.head(5)
    sample_csv = sample_df.to_csv(index=False)

    if answers:
        filled = [a for a in answers if a.get("answer", "").strip()]
        if filled:
            lines = "\n".join(f"  - {a['id']}: {a['answer']}" for a in filled)
            user_context = (
                "User-provided clarifications — treat these as ground truth when writing descriptions:\n"
                f"{lines}\n"
            )
        else:
            user_context = ""
    else:
        user_context = ""

    prompt = RICH_METADATA_PROMPT.format(
        schema=schema_str,
        sample_count=len(sample_df),
        sample_csv=sample_csv,
        user_context=user_context,
    )
    return json.loads(strip_fences(get_response_from_llm(prompt)))


def generate_schema_summary(metadata: dict) -> str:
    lines: list[str] = []
    table = metadata.get("table", {})
    if table.get("description"):
        lines.append(f"Table: {table['description']}")
    if table.get("grain"):
        lines.append(f"Grain: {table['grain']}")
    if table.get("time_range"):
        lines.append(f"Time range: {table['time_range']}")

    columns = metadata.get("columns", [])
    lines.append(f"Columns ({len(columns)}):")
    for col in columns:
        name = col.get("column", "")
        typ = col.get("type", "")
        desc = col.get("description", "")
        aliases = col.get("aliases", [])
        alias_str = f" [also: {', '.join(aliases[:3])}]" if aliases else ""
        lines.append(f"  {name} ({typ}): {desc}{alias_str}")

    return "\n".join(lines)


def generate_smart_parser_questions(df: pd.DataFrame) -> list[dict]:
    from config import SMART_PARSER_QUESTIONS_PROMPT
    schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    sample_df = df.head(5)
    prompt = SMART_PARSER_QUESTIONS_PROMPT.format(
        schema=schema_str,
        sample_count=len(sample_df),
        sample_csv=sample_df.to_csv(index=False),
    )
    return json.loads(strip_fences(get_response_from_llm(prompt)))
