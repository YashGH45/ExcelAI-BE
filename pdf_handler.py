from rank_bm25 import BM25Okapi
from session import Session
from llm import get_response_from_llm
from dto_models import SearchResult


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
        session.pdf_conversation_summary = get_response_from_llm(compress_prompt)
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


def pdf_search(query: str, session: Session, top_k: int = 3) -> SearchResult:
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
    answer = get_response_from_llm(answer_prompt)

    page_refs = ", ".join(f"p.{p}" for p in matched_page_nums)
    _store_pdf_conversation(query, page_refs, answer, session)
    print(f"[pdf-index] answered from pages: {page_refs}")

    return SearchResult(reply=answer)
