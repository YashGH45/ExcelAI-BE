import json
from enum import Enum
from dictionary import (
    _CLEANSE_KEYWORDS, _WRITE_PHRASES, _CHART_KEYWORDS
)
from llm import get_response_from_llm, strip_fences


class QueryIntent(Enum):
    ANSWER_ONLY = "answer_only"
    ANSWER_WITH_CHART = "answer_with_chart"
    ANSWER_WITH_INSIGHTS = "answer_with_insights"
    ANSWER_WITH_CHART_AND_INSIGHTS = "answer_with_chart_and_insights"
    WRITE = "write"
    CLEANSE = "cleanse"


def classify_intent(query: str, schema_summary: str = "") -> QueryIntent:
    q = query.lower().strip()

    if any(kw in q for kw in _CLEANSE_KEYWORDS):
        return QueryIntent.CLEANSE

    if any(phrase in q for phrase in _WRITE_PHRASES):
        return QueryIntent.WRITE

    if any(kw in q for kw in _CHART_KEYWORDS):
        return QueryIntent.ANSWER_WITH_CHART_AND_INSIGHTS

    # Everything else is ambiguous — let the LLM decide
    return classify_intent_with_llm(query, schema_summary)


def classify_intent_with_llm(query: str, schema_summary: str = "") -> QueryIntent:
    from config import INTENT_PROMPT
    prompt = INTENT_PROMPT.format(query=query, schema_summary=schema_summary)
    try:
        raw = strip_fences(get_response_from_llm(prompt))
        data = json.loads(raw)
        return QueryIntent(data["intent"].lower())
    except Exception as e:
        print(f"[intent] LLM classification failed, defaulting to ANSWER_ONLY: {e}")
        return QueryIntent.ANSWER_ONLY
