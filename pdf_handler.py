import os
import re
from rank_bm25 import BM25Okapi
from session import Session
from llm import get_response_from_llm
from dto_models import SearchResult, PDFSource

_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this",
    "from", "have", "has", "been", "what", "when", "where", "which",
    "who", "how", "can", "will", "would", "could", "should", "may", "might",
}


def _compute_highlight_spans(text: str, query: str) -> list[dict]:
    spans = []
    text_lower = text.lower()

    query_words = {
        w for w in re.findall(r'\b\w+\b', query.lower())
        if len(w) > 3 and w not in _STOPWORDS
    }
    for word in query_words:
        for m in re.finditer(re.escape(word), text_lower):
            spans.append({"start": m.start(), "end": m.end()})

    for m in re.finditer(r'\b\d+(?:\.\d+)?%?\b', text):
        spans.append({"start": m.start(), "end": m.end()})

    spans.sort(key=lambda s: s["start"])
    merged = []
    for span in spans:
        if merged and span["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(span)
    return merged


def _extract_section_heading(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 100:
            return line
    return None


def _chunk_pages(pages: list[dict], max_chars: int = 600) -> list[dict]:
    """Split pages into paragraph-sized chunks. Each chunk keeps its page_num."""
    chunks = []
    for page in pages:
        page_num = page["page_num"]
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', page["text"]) if p.strip()]
        for para in paragraphs:
            if len(para) <= max_chars:
                chunks.append({"page_num": page_num, "text": para})
            else:
                # Split long paragraphs by sentence boundary
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current:
                            chunks.append({"page_num": page_num, "text": current})
                        # Hard-split sentences that exceed max_chars on their own
                        while len(sent) > max_chars:
                            chunks.append({"page_num": page_num, "text": sent[:max_chars]})
                            sent = sent[max_chars:]
                        current = sent
                if current:
                    chunks.append({"page_num": page_num, "text": current})
    return chunks


def load_pdf_index(pages: list[dict], session: Session) -> None:
    chunks = _chunk_pages(pages)
    session.pdf_pages = chunks
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    session.pdf_bm25 = BM25Okapi(tokenized)
    print(f"[pdf-index] built BM25 index over {len(chunks)} chunks from {len(pages)} pages")


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


def pdf_search(query: str, session: Session, top_k: int = 5) -> SearchResult:
    if session.pdf_bm25 is None or not session.pdf_pages:
        return SearchResult(reply="No PDF loaded. Please upload a PDF file first.")

    tokens = query.lower().split()
    scores = session.pdf_bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    print(f"[pdf-search] query tokens: {tokens}")
    print(f"[pdf-search] pages indexed: {len(session.pdf_pages)}")
    print(f"[pdf-search] top scores: {[(i, round(float(scores[i]), 4)) for i in top_indices]}")

    if not top_indices:
        return SearchResult(reply="I couldn't find any relevant content in the PDF for your question.")

    top_scores = [float(scores[i]) for i in top_indices]
    score_min = min(top_scores)
    score_max = max(top_scores)
    score_range = score_max - score_min

    context_parts = []
    matched_page_nums = []
    pdf_sources = []
    for idx, raw_score in zip(top_indices, top_scores):
        page = session.pdf_pages[idx]
        context_parts.append(f"[Page {page['page_num']}]\n{page['text']}")
        matched_page_nums.append(str(page["page_num"]))
        normalized_score = round(((raw_score - score_min) / score_range), 2) if score_range > 0 else 1.0
        pdf_sources.append(PDFSource(
            file_name=os.path.basename(session.current_filepath) if session.current_filepath else "",
            page_number=page["page_num"],
            section_heading=_extract_section_heading(page["text"]),
            text=page["text"],
            score=normalized_score,
            highlight_spans=_compute_highlight_spans(page["text"], query),
        ))

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

    return SearchResult(reply=answer, sources=pdf_sources)
