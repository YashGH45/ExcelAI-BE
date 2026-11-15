import re
import duckdb
import pandas as pd
import pdfplumber
from session import Session
import numpy as np
from paddleocr import PPStructure
import pdfplumber


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


# Initialize PPStructure once outside the function so models load only once.
# Set show_log=False to keep your terminal output clean.
layout_engine = PPStructure(table=True, ocr=True, show_log=False)


def _sort_multicolumn_blocks(blocks: list[dict], page_width: float) -> list[dict]:
  """Sorts detected layout blocks in natural multi-column reading order."""
  # Detect if blocks sit side-by-side (two or more columns)
  mid_x = page_width / 2
  left_col = [b for b in blocks if b["bbox"][2] <= mid_x + 20]
  right_col = [b for b in blocks if b["bbox"][0] >= mid_x - 20]

  # If clear left and right columns exist, read left column top-to-bottom, then right
  if len(left_col) >= 2 and len(right_col) >= 2:
    left_col.sort(key=lambda b: b["bbox"][1])
    right_col.sort(key=lambda b: b["bbox"][1])
    other = [b for b in blocks if b not in left_col and b not in right_col]
    other.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return other + left_col + right_col

  # Default top-to-bottom, left-to-right sorting
  return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def load_pdf_pages(filepath: str, ocr_dpi: int = 300) -> list[dict]:
  pages = []

  with pdfplumber.open(filepath) as pdf:
    for i, page in enumerate(pdf.pages):
      word_bboxes = []
      tables = []

      # layout=True preserves visual spacing on digital pages
      text = (page.extract_text(layout=True) or "").strip()

      # Path 1: Digital PDF (has selectable text)
      if len(text) > 20:
        try:
          raw_words = page.extract_words() or []
          word_bboxes = [
              {
                  "text": w["text"],
                  "x0": round(w["x0"], 2),
                  "top": round(w["top"], 2),
                  "x1": round(w["x1"], 2),
                  "bottom": round(w["bottom"], 2),
              }
              for w in raw_words
          ]
        except Exception:
          word_bboxes = []

        try:
          tables = page.extract_tables() or []
        except Exception:
          tables = []

      # Path 2: Scanned / Image PDF (PaddleOCR + Layout/Table detection)
      else:
        pil_image = page.to_image(resolution=ocr_dpi).original
        img_array = np.array(pil_image)

        # Convert BGR/RGB if needed and run PPStructure
        result = layout_engine(img_array)
        scale = 72.0 / ocr_dpi

        # Sort blocks to handle multi-column layouts properly
        sorted_blocks = _sort_multicolumn_blocks(result, pil_image.width)
        extracted_chunks = []

        for block in sorted_blocks:
          block_type = block.get("type", "")
          res = block.get("res", [])

          # Extract tables as HTML strings (great for LLM RAG context)
          if block_type == "table" and isinstance(res, dict):
            html_table = res.get("html", "")
            if html_table:
              tables.append(html_table)
              extracted_chunks.append(html_table)
            continue

          # Extract text lines and bounding boxes
          if isinstance(res, list):
            for line in res:
              line_text = line.get("text", "").strip()
              box = line.get("text_region", [])
              if not line_text or not box:
                continue

              extracted_chunks.append(line_text)

              # Convert 4-point polygon [[x0,y0], [x1,y0], [x1,y1], [x0,y1]] to bbox
              xs = [pt[0] for pt in box]
              ys = [pt[1] for pt in box]
              word_bboxes.append({
                  "text": line_text,
                  "x0": round(min(xs) * scale, 2),
                  "top": round(min(ys) * scale, 2),
                  "x1": round(max(xs) * scale, 2),
                  "bottom": round(max(ys) * scale, 2),
              })

        text = "\n".join(extracted_chunks)

      if not text and not tables:
        continue

      pages.append({
          "page_num": i + 1,
          "text": text,
          "tables": tables,
          "word_bboxes": word_bboxes,
      })

  return pages
