from session import Session
from intent import QueryIntent, classify_intent
from pdf_handler import pdf_search
from cleanse import cleanse_data
from sql_agent import text_to_sql_search
from dto_models import SearchResult


def search(query: str, session: Session, top_k: int = 3) -> SearchResult:
    if session.current_mode == "pdf":
        return pdf_search(query, session, top_k)

    intent = classify_intent(query, session.dataset_summary or "")

    if intent == QueryIntent.CLEANSE:
        return cleanse_data(session)

    return text_to_sql_search(query, session, intent)
