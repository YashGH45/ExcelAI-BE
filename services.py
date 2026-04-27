import pandas as pd
import json
from config import SCHEMA_AND_TEMPLATE_PROMPT
from ai_client import store, chat


def load_dataframe(filepath: str) -> pd.DataFrame:
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    return df.dropna(how="all")


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


