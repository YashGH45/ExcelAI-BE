import re
import pandas as pd
import json
import pdfplumber
from config import SCHEMA_AND_TEMPLATE_PROMPT
from ai_client import store, chat


def _to_snake_case(name: str) -> str:
    name = str(name).strip()
    # Insert underscore before uppercase letters that follow a lowercase letter or digit (CamelCase)
    name = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name)
    # Replace spaces, hyphens, and other separators with underscores
    name = re.sub(r'[\s\-]+', '_', name)
    name = name.lower()
    # Replace any character that is not alphanumeric or underscore with underscore
    name = re.sub(r'[^\w]', '_', name)
    # Collapse multiple underscores and strip leading/trailing ones
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def _normalize_headers_inplace(filepath: str, df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case and overwrite the original file."""
    df = df.rename(columns={col: _to_snake_case(col) for col in df.columns})
    if filepath.endswith(".csv"):
        df.to_csv(filepath, index=False)
    else:
        df.to_excel(filepath, index=False)
    return df


def load_dataframe(filepath: str) -> pd.DataFrame:
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    df = df.dropna(how="all")
    df = _normalize_headers_inplace(filepath, df)
    return df


def generate_schema_and_template(df: pd.DataFrame) -> dict:
    prompt = SCHEMA_AND_TEMPLATE_PROMPT + "\n" + extract_headers_row(df)
    response = chat(prompt)
    # Strip markdown code fences that LLMs sometimes add despite instructions
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]  # remove opening fence line
        cleaned = cleaned.rsplit("```", 1)[0]  # remove closing fence
    return json.loads(cleaned.strip())


def extract_headers_row(df: pd.DataFrame) -> str:
    return df.head(2).to_csv(index=False)


def load_pdf_pages(filepath: str) -> list[dict]:
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page_num": i + 1, "text": text})
    return pages


def convert_rows_to_sentences(df: pd.DataFrame, template: str) -> None:
    sentences = []
    sentence_ids = []

    for index, row in df.iterrows():
        try:
            sentences.append(template.format(**row))
            sentence_ids.append(f"record_{index}")
        except KeyError as e:
            print(f"Skipping row {index}, missing key: {e}")

    store(sentences, sentence_ids)


