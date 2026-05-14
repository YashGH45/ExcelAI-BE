import time
import ollama
from session import Session
from llm import get_response_from_llm


def _embed_for_memory(text: str) -> list[float] | None:
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"[memory] Ollama unavailable, skipping embedding: {e}")
        return None


def store_conversation(query: str, sql: str, answer: str, session: Session) -> None:
    if len(session.recent_turns) == session.recent_turns.maxlen:
        oldest_q, oldest_sql, oldest_a = session.recent_turns[0]
        compress_prompt = (
            (f"Existing summary:\n{session.conversation_summary}\n\n" if session.conversation_summary else "") +
            f"New turn to fold in:\nQ: {oldest_q}\nA: {oldest_a}\n\n"
            "Update the summary in 2-3 sentences preserving key facts, column names, "
            "filter values, and numeric results. Return only the updated summary text, nothing else."
        )
        session.conversation_summary = get_response_from_llm(compress_prompt)
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


def format_recent_turns(session: Session) -> str:
    if not session.recent_turns:
        return ""
    lines = []
    for i, (q, s, a) in enumerate(session.recent_turns):
        label = "MOST RECENT" if i == len(session.recent_turns) - 1 else f"Turn {i + 1}"
        lines.append(f"[{label}]\nQ: {q}\nSQL: {s}\nA: {a}")
    return "\n\n".join(lines)


def retrieve_similar_conversations(query: str, session: Session, top_k: int = 3) -> str:
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
