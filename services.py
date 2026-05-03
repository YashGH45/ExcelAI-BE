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


def generate_smart_parser_questions(df: pd.DataFrame) -> list[dict]:
    """Ask the LLM what it's uncertain about so the user can clarify before metadata generation."""
    from config import SMART_PARSER_QUESTIONS_PROMPT
    schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    sample_df = df.head(5)
    prompt = SMART_PARSER_QUESTIONS_PROMPT.format(
        schema=schema_str,
        sample_count=len(sample_df),
        sample_csv=sample_df.to_csv(index=False),
    )
    response = chat(prompt)
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())


def generate_rich_metadata(df: pd.DataFrame, answers: list[dict] | None = None) -> dict:
    """Generate rich table + column metadata via a single LLM call.

    answers: optional list of {"id": str, "answer": str} from the smart-parser
             clarification step. When provided, the LLM incorporates them into
             the metadata instead of having to guess.
    """
    from config import RICH_METADATA_PROMPT
    schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    sample_df = df.head(5)
    sample_csv = sample_df.to_csv(index=False)

    if answers:
        filled = [a for a in answers if a.get("answer", "").strip()]
        if filled:
            lines = "\n".join(f"  - {a['id']}: {a['answer']}" for a in filled)
            user_context = (
                "User-provided clarifications — treat these as ground truth when writing descriptions:\n"
                f"{lines}\n"
            )
        else:
            user_context = ""
    else:
        user_context = ""

    prompt = RICH_METADATA_PROMPT.format(
        schema=schema_str,
        sample_count=len(sample_df),
        sample_csv=sample_csv,
        user_context=user_context,
    )
    response = chat(prompt)
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())


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


