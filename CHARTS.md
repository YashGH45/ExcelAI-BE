# Vega-Lite Charts & Insights — Implementation Guide

## Overview

Every chat response on the Excel/CSV path now carries two optional fields alongside
the existing `reply` text:

| Field | Type | Description |
|---|---|---|
| `vega_spec` | `object \| null` | Complete Vega-Lite v5 JSON spec, ready for `vega-embed` |
| `insights` | `string[]` | 2–4 data-backed analytical observations |

PDFs return `vega_spec: null` and `insights: []` — charts are only generated from
structured tabular results.

---

## End-to-End Workflow

```
POST /api/chat  { "message": "Show monthly revenue by category" }
│
├─ app.py  →  search(message)
│
├─ ai_client.py — _text_to_sql_search(query)
│   │
│   ├─ 1. DESCRIBE schema from DuckDB
│   ├─ 2. Build memory/context block (recent turns + vector fallback)
│   ├─ 3. LLM → generate SQL  (groq llama-3.3-70b-versatile)
│   ├─ 4. Execute SQL via DuckDB  (agentic self-heal loop, up to 5 retries)
│   ├─ 5. Truncate result to MAX_RESULT_ROWS (20)
│   │
│   ├─ 6. LLM call A — natural-language summary  (summary_prompt)
│   │       → result.reply
│   │
│   ├─ 7. LLM call B — chart + insights  (_generate_chart_and_insights)
│   │       → sends: query + schema + full data rows (JSON)
│   │       → receives: { "vega_spec": {...}|null, "insights": [...] }
│   │       → result.vega_spec, result.insights
│   │
│   └─ 8. Store Q+SQL+A in conversation memory (ChromaDB + deque)
│
└─ ChatResponse { reply, vega_spec, insights }
```

---

## API Contract

### Request
```json
POST /api/chat
{
  "message": "Show total bookings per hotel type"
}
```

### Response
```json
{
  "reply": "There were 79,330 City Hotel bookings and 40,060 Resort Hotel bookings...",

  "vega_spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": "Total Bookings by Hotel Type",
    "width": "container",
    "height": 320,
    "data": {
      "values": [
        { "hotel": "City Hotel",   "bookings": 79330 },
        { "hotel": "Resort Hotel", "bookings": 40060 }
      ]
    },
    "mark": "bar",
    "encoding": {
      "x": { "field": "hotel",    "type": "nominal",      "axis": { "title": "Hotel Type" } },
      "y": { "field": "bookings", "type": "quantitative", "axis": { "title": "Total Bookings" } },
      "tooltip": true
    }
  },

  "insights": [
    "City Hotel accounts for 66% of total bookings — nearly twice the Resort Hotel volume",
    "Combined bookings across both types total 119,390"
  ]
}
```

When the query produces a single scalar, a text answer, or fewer than 2 rows the
response is:
```json
{ "reply": "...", "vega_spec": null, "insights": [] }
```

---

## Chart Type Decision Logic

The LLM selects the mark type automatically based on the data shape:

| Data pattern | Vega-Lite mark |
|---|---|
| Categorical X + numeric Y | `"bar"` |
| Temporal / ordered X + numeric Y | `"line"` |
| Two numeric axes | `"point"` (scatter) |
| Proportions, ≤ 8 categories | `"arc"` (pie/donut) |
| Cumulative / filled time series | `"area"` |
| > 6 categories on X-axis | `"bar"` with flipped axes |

Spec is `null` when:
- Result has fewer than 2 rows
- Result is a single scalar (one COUNT / one SUM)
- No numeric column is present for a meaningful axis

---

## Key Files Changed

| File | Change |
|---|---|
| `requirements.txt` | Added `altair` |
| `config.py` | Added `CHART_AND_INSIGHTS_PROMPT` |
| `dto_models.py` | Added `vega_spec: dict \| None` and `insights: list[str]` to `ChatResponse` |
| `ai_client.py` | Added `SearchResult` dataclass; added `_generate_chart_and_insights()`; updated all return types through `search()` |
| `app.py` | Updated `chat_endpoint` to forward `vega_spec` and `insights` |

---

## Frontend Integration

Render with [vega-embed](https://github.com/vega/vega-embed):

```js
import embed from 'vega-embed'

// After receiving the API response:
if (response.vega_spec) {
  embed('#chart-container', response.vega_spec, {
    actions: { export: true, source: false, compiled: false, editor: false },
    renderer: 'svg',
  })
}

// Render insights as a bullet list:
response.insights.forEach(text => {
  const li = document.createElement('li')
  li.textContent = text
  insightsList.appendChild(li)
})
```

The spec already contains all data inline (`"data": { "values": [...] }`) — no
extra fetch is required.

---

## Scope & Limitations

| Scope | Detail |
|---|---|
| Excel / CSV only | PDF mode always returns `vega_spec: null, insights: []` |
| Max rows in chart | 20 rows (controlled by `MAX_RESULT_ROWS`) |
| LLM model | Same `groq llama-3.3-70b-versatile` used for all calls |
| Extra latency | One additional LLM call per SQL query (chart + insights combined) |
| Spec validity | LLM output is parsed as JSON; parse failures silently return nulls |
| No Altair dependency | Spec is generated directly by the LLM — `altair` is available for future use |