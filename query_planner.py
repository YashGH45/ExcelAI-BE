import json

from llm import get_response_from_llm, strip_fences

_AGGREGATION_OPS = {"SUM", "COUNT", "AVG", "MAX", "MIN"}


def detect_sql_intent(query: str, schema_str: str, metadata_section: str = "") -> dict:
    """
    Stage A — SQL Intent Detection.

    Analyzes the user query against the schema and returns a structured intent dict.

    Returns:
      - Intent dict with operations, target_columns, filters, etc.
      - {"ambiguous": True, "clarification": "..."} to short-circuit the pipeline.
      - {} on any LLM/parse failure (graceful degradation — pipeline continues without a plan).
    """
    from config import INTENT_DETECTION_PROMPT

    prompt = INTENT_DETECTION_PROMPT.format(
        query=query,
        schema_str=schema_str,
        metadata_section=metadata_section,
    )
    try:
        raw = strip_fences(get_response_from_llm(prompt))
        result = json.loads(raw)
        if not isinstance(result, dict):
            print("[intent-detect] LLM returned non-dict — skipping")
            return {}
        if result.get("ambiguous"):
            print("[intent-detect] ambiguous query detected")
        else:
            print(f"[intent-detect] operations={result.get('operations')}  output_type={result.get('output_type')}")
        return result
    except Exception as exc:
        print(f"[intent-detect] failed (non-fatal): {exc}")
        return {}


def build_query_plan(intent: dict, query: str, schema_str: str) -> dict:
    """
    Stage B — Query Planning.

    Consumes the intent dict from detect_sql_intent() and produces a structured
    execution plan (select_columns, aggregations, filters, group_by, order_by, etc.)
    that _generate_sql() will use in place of the raw query alone.

    Returns:
      - Plan dict on success.
      - {} on any LLM/parse failure or validation mismatch (graceful degradation).
    """
    from config import QUERY_PLANNING_PROMPT

    try:
        intent_json = json.dumps(intent, indent=2)
    except Exception:
        intent_json = str(intent)

    prompt = QUERY_PLANNING_PROMPT.format(
        query=query,
        schema_str=schema_str,
        intent_json=intent_json,
    )
    try:
        raw = strip_fences(get_response_from_llm(prompt))
        plan = json.loads(raw)
        if not isinstance(plan, dict):
            print("[query-plan] LLM returned non-dict — skipping")
            return {}

        # Validate: aggregation-requiring intents must have at least one aggregation.
        operations = {op.upper() for op in intent.get("operations", [])}
        if operations & _AGGREGATION_OPS and not plan.get("aggregations"):
            print(
                "[query-plan] validation failed: intent requires aggregation but plan "
                "has none — skipping plan"
            )
            return {}

        print(f"[query-plan] plan_summary={plan.get('plan_summary', '(none)')!r}")
        return plan
    except Exception as exc:
        print(f"[query-plan] failed (non-fatal): {exc}")
        return {}
