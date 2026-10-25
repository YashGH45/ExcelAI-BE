import os
import re
from rank_bm25 import BM25Okapi
from session import Session
from llm import get_response_from_llm
from dto_models import SearchResult, PDFSource, PDFEvidence

_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this",
    "from", "have", "has", "been", "what", "when", "where", "which",
    "who", "how", "can", "will", "would", "could", "should", "may", "might",
}


def _extract_used_page_numbers(answer: str) -> set[int]:
    """Parse [Page N] citations from the LLM answer to find which pages were actually used."""
    return {int(m.group(1)) for m in re.finditer(r'\[Page\s+(\d+)]', answer, re.IGNORECASE)}


def _extract_answer_keywords(answer: str) -> set[str]:
    """Meaningful content words from the LLM answer, used to score source sentences."""
    tokens = set(re.findall(r'\b\w{4,}\b', answer.lower()))
    return tokens - _STOPWORDS


def _extract_used_sentences(chunk_text: str, answer_keywords: set[str], threshold: float = 0.2) -> list[dict]:
    """
    Score each line/sentence in chunk_text by word-overlap with answer keywords.
    Returns [{start, end}] offsets into chunk_text for lines scoring >= threshold.
    """
    if not answer_keywords:
        return []

    spans = []
    pos = 0
    for raw_line in chunk_text.split('\n'):
        stripped = raw_line.strip()
        line_len = len(raw_line)

        if len(stripped) >= 15:
            lpad = raw_line.index(stripped[0])
            line_start = pos + lpad

            if len(stripped) <= 250:
                words = set(re.findall(r'\b\w{4,}\b', stripped.lower())) - _STOPWORDS
                if words and len(words & answer_keywords) / len(words) >= threshold:
                    spans.append({"start": line_start, "end": line_start + len(stripped)})
            else:
                # Long paragraph: score sentence by sentence
                for m in re.finditer(r'[^.!?]+[.!?]?', stripped):
                    s = m.group().strip()
                    if len(s) < 15:
                        continue
                    words = set(re.findall(r'\b\w{4,}\b', s.lower())) - _STOPWORDS
                    if words and len(words & answer_keywords) / len(words) >= threshold:
                        abs_start = line_start + m.start()
                        spans.append({"start": abs_start, "end": abs_start + len(s)})

        pos += line_len + 1  # +1 for the '\n'

    return spans


def _build_cited_snippet(chunk_text: str, sentence_spans: list[dict]) -> tuple[str, list[dict]]:
    """
    Assemble a snippet from the used sentences only.
    Returns (snippet_text, highlight_spans) where each span covers an entire sentence.
    """
    sentence_spans = sorted(sentence_spans, key=lambda s: s["start"])
    parts: list[str] = []
    highlight_spans: list[dict] = []
    cursor = 0
    prev_end = -1

    for span in sentence_spans:
        sentence_text = chunk_text[span["start"]:span["end"]].strip()
        if not sentence_text:
            continue
        if prev_end != -1:
            gap = " … " if span["start"] > prev_end else " "
            parts.append(gap)
            cursor += len(gap)
        highlight_spans.append({"start": cursor, "end": cursor + len(sentence_text)})
        parts.append(sentence_text)
        cursor += len(sentence_text)
        prev_end = span["end"]

    return "".join(parts), highlight_spans


def _union_bbox(bboxes: list[dict]) -> dict | None:
    """Compute the bounding box that contains all given word bboxes."""
    if not bboxes:
        return None
    return {
        "x0": min(b["x0"] for b in bboxes),
        "top": min(b["top"] for b in bboxes),
        "x1": max(b["x1"] for b in bboxes),
        "bottom": max(b["bottom"] for b in bboxes),
    }


def _bbox_for_chunk(chunk_text: str, word_bboxes: list[dict], word_cursor: list[int]) -> dict | None:
    """
    Match chunk_text words against page word_bboxes starting from word_cursor[0].
    Advances word_cursor[0] past the matched region.
    Returns union bbox of matched words, or None on failure (scanned PDF, mismatch).
    """
    if not word_bboxes or not chunk_text.strip():
        return None
    chunk_words = [w for w in re.split(r'\s+', chunk_text.strip()) if w]
    if not chunk_words:
        return None

    matched: list[dict] = []
    i = word_cursor[0]
    ci = 0
    max_scan = len(chunk_words) * 3 + 10  # slack for OCR/punctuation differences

    while ci < len(chunk_words) and i < len(word_bboxes) and (i - word_cursor[0]) < max_scan:
        if word_bboxes[i]["text"] == chunk_words[ci]:
            matched.append(word_bboxes[i])
            ci += 1
        i += 1  # always advance page-word pointer

    if matched:
        word_cursor[0] = i
    return _union_bbox(matched)


def _extract_section_heading(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 100:
            return line
    return None


def _chunk_pages(pages: list[dict], max_chars: int = 600) -> list[dict]:
    """
    Split pages into paragraph-sized chunks.
    Each chunk carries: chunk_id, page_num, paragraph_id, heading,
    text, char_start, char_end (offsets into page text), bbox.
    """
    chunks = []
    for page in pages:
        page_num = page["page_num"]
        page_text = page["text"]
        word_bboxes = page.get("word_bboxes", [])
        word_cursor = [0]   # advances through word_bboxes in reading order
        page_cursor = 0     # advances through page_text to locate paragraphs

        para_idx = 0
        for raw_para in re.split(r'\n{2,}', page_text):
            para = raw_para.strip()
            if not para:
                continue

            para_start = page_text.find(para, page_cursor)
            if para_start == -1:
                para_start = page_cursor
            page_cursor = para_start + len(para)
            heading = _extract_section_heading(para)

            if len(para) <= max_chars:
                chunks.append({
                    "chunk_id": f"p{page_num}_para{para_idx}_c0",
                    "page_num": page_num,
                    "paragraph_id": para_idx,
                    "heading": heading,
                    "text": para,
                    "char_start": para_start,
                    "char_end": para_start + len(para),
                    "bbox": _bbox_for_chunk(para, word_bboxes, word_cursor),
                })
            else:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                current_char_start = para_start
                chunk_seq = 0
                sent_cursor_in_para = 0

                for sent in sentences:
                    sent_pos = para.find(sent, sent_cursor_in_para)
                    if sent_pos == -1:
                        sent_pos = sent_cursor_in_para

                    if len(current) + len(sent) + 1 <= max_chars:
                        if not current:
                            current_char_start = para_start + sent_pos
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current:
                            chunks.append({
                                "chunk_id": f"p{page_num}_para{para_idx}_c{chunk_seq}",
                                "page_num": page_num,
                                "paragraph_id": para_idx,
                                "heading": heading,
                                "text": current,
                                "char_start": current_char_start,
                                "char_end": current_char_start + len(current),
                                "bbox": _bbox_for_chunk(current, word_bboxes, word_cursor),
                            })
                            chunk_seq += 1

                        abs_sent_start = para_start + sent_pos
                        while len(sent) > max_chars:
                            chunks.append({
                                "chunk_id": f"p{page_num}_para{para_idx}_c{chunk_seq}",
                                "page_num": page_num,
                                "paragraph_id": para_idx,
                                "heading": heading,
                                "text": sent[:max_chars],
                                "char_start": abs_sent_start,
                                "char_end": abs_sent_start + max_chars,
                                "bbox": _bbox_for_chunk(sent[:max_chars], word_bboxes, word_cursor),
                            })
                            chunk_seq += 1
                            abs_sent_start += max_chars
                            sent = sent[max_chars:]

                        current = sent
                        current_char_start = abs_sent_start

                    sent_cursor_in_para = sent_pos + len(sent)

                if current:
                    chunks.append({
                        "chunk_id": f"p{page_num}_para{para_idx}_c{chunk_seq}",
                        "page_num": page_num,
                        "paragraph_id": para_idx,
                        "heading": heading,
                        "text": current,
                        "char_start": current_char_start,
                        "char_end": current_char_start + len(current),
                        "bbox": _bbox_for_chunk(current, word_bboxes, word_cursor),
                    })

            para_idx += 1

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

    context_parts = []
    matched_page_nums = []
    retrieved_chunks = []
    for idx in top_indices:
        page = session.pdf_pages[idx]
        context_parts.append(f"[Page {page['page_num']}]\n{page['text']}")
        matched_page_nums.append(str(page["page_num"]))
        retrieved_chunks.append({
            "chunk_id":     page.get("chunk_id", ""),
            "page_num":     page["page_num"],
            "paragraph_id": page.get("paragraph_id", 0),
            "heading":      page.get("heading"),
            "text":         page["text"],
            "char_start":   page.get("char_start", 0),
            "char_end":     page.get("char_end", 0),
            "bbox":         page.get("bbox"),
            "score":        round(float(scores[idx]), 4),
        })

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

    used_pages = _extract_used_page_numbers(answer)
    answer_keywords = _extract_answer_keywords(answer)

    file_name = os.path.basename(session.current_filepath) if session.current_filepath else ""
    pdf_sources: list[PDFSource] = []
    pdf_evidence: list[PDFEvidence] = []

    for chunk in retrieved_chunks:
        if used_pages and chunk["page_num"] not in used_pages:
            continue
        sentence_spans = _extract_used_sentences(chunk["text"], answer_keywords)
        if not sentence_spans:
            continue
        snippet, highlight_spans = _build_cited_snippet(chunk["text"], sentence_spans)

        pdf_sources.append(PDFSource(
            file_name=file_name,
            page_number=chunk["page_num"],
            section_heading=chunk["heading"],
            text=snippet,
            score=chunk["score"],
            highlight_spans=highlight_spans,
        ))
        pdf_evidence.append(PDFEvidence(
            chunk_id=chunk["chunk_id"],
            page=chunk["page_num"],
            heading=chunk["heading"],
            paragraph_ids=[chunk["paragraph_id"]],
            bounding_boxes=[chunk["bbox"]] if chunk["bbox"] else [],
            character_offsets={"start": chunk["char_start"], "end": chunk["char_end"]},
            similarity_score=chunk["score"],
            evidence_text=snippet or None,
        ))

    # Sort by BM25 score, cap at 5; keep sources and evidence in sync
    combined = sorted(zip(pdf_sources, pdf_evidence), key=lambda p: p[0].score, reverse=True)[:5]
    pdf_sources = [s for s, _ in combined]
    pdf_evidence = [e for _, e in combined]

    page_refs = ", ".join(f"p.{p}" for p in matched_page_nums)
    _store_pdf_conversation(query, page_refs, answer, session)
    print(f"[pdf-index] answered from pages: {page_refs} | cited: {used_pages or 'none'} | sources built: {len(pdf_sources)}")

    return SearchResult(reply=answer, sources=pdf_sources, evidence=pdf_evidence)
