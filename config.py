from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


CHART_AND_INSIGHTS_PROMPT = """\
You are a data visualization and analytics expert.

User query: {query}

Column names and dtypes:
{schema}

Data rows (JSON array):
{data_json}

Return a single JSON object with exactly two keys:

{{
  "vega_spec": <Vega-Lite v5 spec object> or null,
  "insights": ["insight 1", "insight 2", ...]
}}

─── vega_spec rules ───────────────────────────────────────────────────────────
• Set to null when:
    - fewer than 2 data rows
    - result is a single scalar value (one COUNT / one total)
    - no numeric column exists for a meaningful axis
• "$schema": "https://vega.github.io/schema/vega-lite/v5.json"
• Embed data inline:  "data": {{"values": [...]}}
• "width": "container",  "height": 320
• Include a concise "title" derived from the query
• "tooltip": true  (or explicit per-field encoding)
• Mark type selection:
    "bar"   → categorical/nominal X + quantitative Y
    "line"  → temporal or ordered X + quantitative Y
    "point" → two quantitative axes (scatter)
    "arc"   → proportions, ≤ 8 distinct categories (pie/donut)
    "area"  → cumulative / filled time series
• For bar charts with > 6 categories: flip axes (x = quantitative, y = nominal)
• For "arc": use "theta" (quantitative) + "color" (nominal) encoding

─── insights rules ────────────────────────────────────────────────────────────
• 2–4 concise, data-backed analytical observations about the result
• Each insight must reference a concrete number, percentage, rank, or trend
• Examples:
    "Revenue grew 23% month-over-month in March"
    "Electronics accounted for 40% of total sales — the highest category"
• If the result is too sparse for meaningful insights, return an empty array []

Return ONLY the raw JSON object. No markdown fences, no explanation.\
"""


CHART_ONLY_PROMPT = """\
You are a data visualization expert.

User query: {query}

Column names and dtypes:
{schema}

Data rows (JSON array):
{data_json}

Return a single JSON object with exactly two keys:

{{
  "vega_spec": <Vega-Lite v5 spec object> or null,
  "insights": []
}}

─── vega_spec rules ───────────────────────────────────────────────────────────
• Set to null when:
    - fewer than 2 data rows
    - result is a single scalar value (one COUNT / one total)
    - no numeric column exists for a meaningful axis
• "$schema": "https://vega.github.io/schema/vega-lite/v5.json"
• Embed data inline:  "data": {{"values": [...]}}
• "width": "container",  "height": 320
• Include a concise "title" derived from the query
• "tooltip": true  (or explicit per-field encoding)
• Mark type selection:
    "bar"   → categorical/nominal X + quantitative Y
    "line"  → temporal or ordered X + quantitative Y
    "point" → two quantitative axes (scatter)
    "arc"   → proportions, ≤ 8 distinct categories (pie/donut)
    "area"  → cumulative / filled time series
• For bar charts with > 6 categories: flip axes (x = quantitative, y = nominal)
• For "arc": use "theta" (quantitative) + "color" (nominal) encoding

Always set "insights" to an empty array [].

Return ONLY the raw JSON object. No markdown fences, no explanation.\
"""


INSIGHTS_ONLY_PROMPT = """\
You are a data analytics expert.

User query: {query}

Column names and dtypes:
{schema}

Data rows (JSON array):
{data_json}

Return a single JSON object with exactly two keys:

{{
  "vega_spec": null,
  "insights": ["insight 1", "insight 2", ...]
}}

─── insights rules ────────────────────────────────────────────────────────────
• 2–4 concise, data-backed analytical observations about the result
• Each insight must reference a concrete number, percentage, rank, or trend
• Examples:
    "Revenue grew 23% month-over-month in March"
    "Electronics accounted for 40% of total sales — the highest category"
• If the result is too sparse for meaningful insights, return an empty array []

Always set "vega_spec" to null.

Return ONLY the raw JSON object. No markdown fences, no explanation.\
"""


CLEANSE_PROMPT = """\
You are a data cleansing expert. Your task is to fix dirty rows in a dataset.

Dataset schema (column name → dtype):
{schema}

Reference rows — clean, complete examples that show the expected values and formats:
{reference_rows}

Dirty rows to fix — may contain null values, placeholder strings like "N/A" / "null" / "unknown", \
type mismatches, or inconsistent formatting:
{dirty_rows}

Rules:
1. Infer correct values from the schema and reference rows for context.
2. Fix null values, null-like strings, type mismatches, and formatting inconsistencies.
3. If a value truly cannot be inferred, keep it as null.
4. Preserve the "_row_index" field exactly as provided — it is required for mapping corrections back.
5. Return ONLY the corrected dirty rows — do NOT include the reference rows.
6. Do NOT change fields that are already correct.

Return ONLY a raw JSON array of the corrected rows. No explanation, no markdown, no extra text.\
"""


RICH_METADATA_PROMPT = """\
You are a data analyst generating rich metadata for a dataset.

Dataset schema (column name → dtype):
{schema}

Sample rows (first {sample_count} rows):
{sample_csv}

Generate a comprehensive metadata object. Return a single JSON object:

{{
  "table": {{
    "description": "What this table represents (1-2 sentences)",
    "grain": "One row = one [entity] (e.g. one order, one guest stay, one product)",
    "time_range": "Approximate time range of the data, or null if no date column exists",
    "caveats": ["known data quality issue or limitation if detectable from sample, else empty"]
  }},
  "columns": [
    {{
      "column": "<exact column name as it appears in the schema>",
      "type": "<inferred type: INTEGER, FLOAT, TEXT, DATE, BOOLEAN, CATEGORICAL>",
      "description": "<what this column represents, including units if numeric>",
      "example_values": [<3-5 representative values drawn from the sample rows>],
      "aliases": ["<alternate names a non-technical user might say to refer to this column>"],
      "related_columns": ["<other column names from the schema that are semantically linked>"]
    }}
  ]
}}

Rules:
- Include exactly one entry per column in the schema — no omissions, no extras
- For aliases, think about how a business user might phrase the column (e.g. "revenue" → ["income", "earnings", "money made", "total sales"])
- For related_columns, only reference column names that actually exist in the provided schema
- Use CATEGORICAL type when a column is likely to have fewer than 20 distinct values
- For time_range, inspect the sample values to infer an approximate range (e.g. "2022–2024"); use null if no date column exists
- If caveats cannot be detected from the sample, return an empty array []
{user_context}\
- Return ONLY the raw JSON object. No markdown fences, no explanation.\
"""


SMART_PARSER_QUESTIONS_PROMPT = """\
You are a data analyst examining a new dataset. Your goal is to ask the user questions/doubts regarding the dataset you are Unsure about \
whose answers will help you generate rich, accurate metadata.
you may ask at Max. 7 questions

Dataset schema (column name → dtype):
{schema}

Sample rows (first {sample_count} rows):
{sample_csv}

Identify the things you are genuinely UNSURE about — things that cannot be reliably inferred from column names \
and the sample values alone. Good question topics:
- Business meaning or units of a numeric column (e.g. is "amount" in USD, EUR, thousands?)
- What a status code or enum value means (e.g. status=1 vs status=2)
- Grain of the data (one row = one what?)
- Whether a column is a raw value or a derived/calculated field
- Known caveats or data quality limitations
- Time period the data covers, if unclear from sample dates
- Whether an ID column is a foreign key to another system

Do NOT ask about things that are already self-evident from the column name and sample values.

Return a JSON array of question objects — maximum 7 items:
[
  {{
    "id": "<short_snake_case_identifier unique to this question>",
    "label": "<clear, plain-English question for the user>",
    "placeholder": "<brief example answer or hint, e.g. 'One row = one hotel booking'>",
    "required": true
  }}
]

Return ONLY the raw JSON array. No markdown fences, no explanation.\
"""


PDF_ANSWER_PROMPT = """\
You are a helpful assistant answering questions about a PDF document.

Retrieved PDF pages:
{context}

{history}\
Question: {query}

Answer based solely on the provided page content. Cite page numbers where relevant \
(e.g. "According to page 3..."). If the answer cannot be found in the retrieved pages, \
say so clearly instead of guessing.
Be concise and accurate.\
"""

INTENT_PROMPT = """
Classify this query into one of: CLEANSE, WRITE, ANSWER_WITH_CHART, 
ANSWER_WITH_INSIGHTS, ANSWER_WITH_CHART_AND_INSIGHTS, ANSWER_ONLY.

Query: {query}
Schema context: {schema_summary}

Return JSON: {{"intent": "...", "confidence": 0.0-1.0}}
"""


SCHEMA_AND_TEMPLATE_PROMPT = """
You are a data analyst. Given the headers and sample rows of a spreadsheet, return a JSON object with exactly 2 properties:

1. "template": A single natural language sentence using {{column_name}} placeholders that describes one record clearly.
2. "summary": The description should cover:
- What the dataset is about overall
- What each column represents in plain English, including its data type and what kind of values it holds
- Any columns that appear categorical, mentioning their likely possible values
- The general purpose or use case of this data (e.g. sales tracking, employee records, inventory management)
- Any relationships between columns that are obvious (e.g. order_date and status likely relate to order lifecycle)

Rules:
- Write in flowing prose, not bullet points or structured format
- Be specific — mention actual column names and sample values naturally in the text
- Identify nature of column and generate necessary & Information packed description
- For numeric columns, find max, mean, avg and n frequent values 
- Avoid generic filler phrases like "this dataset contains information about"
- The output will be embedded as a vector, so pack it with meaningful descriptive terms
- Return only the description, no preamble, no headings, no extra formatting
- Infer data types from the sample values, not just the header name
- If a column looks like a date, label it as date
- If a column has only a few possible values (like status, region), flag it as categorical
- Return only valid JSON, no preamble, no markdown, no backticks
"""