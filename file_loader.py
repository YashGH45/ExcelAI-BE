import re
import duckdb
import pandas as pd
import pdfplumber
from session import Session



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


def load_dataframe(filepath: str) -> tuple[pd.DataFrame, str | None]:
    sheet_name = None
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        xf = pd.ExcelFile(filepath)
        sheet_name = xf.sheet_names[0] if xf.sheet_names else None
        df = xf.parse(sheet_name or 0)
    df = df.dropna(how="all")
    df = _normalize_headers_inplace(filepath, df)
    return df, sheet_name


def load_into_duckdb(df: pd.DataFrame, session: Session) -> None:
    session.duck_conn = duckdb.connect()
    session.duck_conn.register("data", df)


def load_pdf_pages(filepath: str) -> list[dict]:
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            # Extract word-level bounding boxes for evidence support.
            # Falls back to [] for scanned/image PDFs where extract_words returns nothing.
            try:
                raw_words = page.extract_words() or []
                word_bboxes = [
                    {"text": w["text"], "x0": w["x0"], "top": w["top"], "x1": w["x1"], "bottom": w["bottom"]}
                    for w in raw_words
                ]
            except Exception:
                word_bboxes = []
            pages.append({"page_num": i + 1, "text": text, "word_bboxes": word_bboxes})
    return pages
