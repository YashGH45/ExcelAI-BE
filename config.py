from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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

