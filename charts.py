import json
import pandas as pd
from llm import get_response_from_llm, strip_fences


def generate_chart_and_insights(
    df: pd.DataFrame,
    query: str,
    want_chart: bool = True,
    want_insights: bool = True,
) -> tuple[dict | None, list[str]]:
    if df is None or df.empty or (not want_chart and not want_insights):
        return None, []

    schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    data_json = df.to_json(orient="records", date_format="iso")

    from config import CHART_AND_INSIGHTS_PROMPT, CHART_ONLY_PROMPT, INSIGHTS_ONLY_PROMPT

    if want_chart and want_insights:
        prompt = CHART_AND_INSIGHTS_PROMPT.format(query=query, schema=schema_str, data_json=data_json)
    elif want_chart:
        prompt = CHART_ONLY_PROMPT.format(query=query, schema=schema_str, data_json=data_json)
    else:
        prompt = INSIGHTS_ONLY_PROMPT.format(query=query, schema=schema_str, data_json=data_json)

    raw = strip_fences(get_response_from_llm(prompt))

    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None, []

        spec = payload.get("vega_spec") if want_chart else None
        if not isinstance(spec, dict):
            spec = None

        insights = payload.get("insights", []) if want_insights else []
        if not isinstance(insights, list):
            insights = []
        insights = [str(i) for i in insights]

        if spec:
            print(f"[chart] Vega-Lite spec generated  |  {len(insights)} insight(s)")
        elif want_insights:
            print(f"[chart] {len(insights)} insight(s) generated (no chart)")
        return spec, insights

    except json.JSONDecodeError as exc:
        print(f"[chart] invalid JSON from LLM — skipping: {exc}")
        return None, []
