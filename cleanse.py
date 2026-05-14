import os
import json
import pandas as pd
from session import Session
from llm import get_response_from_llm, strip_fences
from dto_models import SearchResult

_NULL_LIKE_STRINGS = frozenset({
    "n/a", "na", "null", "none", "nan", "#n/a", "#na", "-", "--",
    "?", "unknown", "missing", "undefined", "nil", "void", "",
})

MAX_DIRTY_BATCH = 50


def _identify_dirty_rows(df: pd.DataFrame) -> pd.Index:
    dirty_mask = pd.Series(False, index=df.index)

    dirty_mask |= df.isnull().any(axis=1)

    for col in df.select_dtypes(include="object").columns:
        col_lower = df[col].astype(str).str.lower().str.strip()
        dirty_mask |= col_lower.isin(_NULL_LIKE_STRINGS)

    for col in df.select_dtypes(include="object").columns:
        numeric_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
        if numeric_count / len(df) >= 0.7:
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            not_null_like = ~df[col].astype(str).str.lower().str.strip().isin(_NULL_LIKE_STRINGS)
            dirty_mask |= non_numeric & not_null_like

    return df.index[dirty_mask]


def cleanse_data(session: Session) -> SearchResult:
    if session.duck_conn is None:
        return SearchResult(reply="No data loaded. Please upload a file first.")
    if not session.current_filepath:
        return SearchResult(reply="Source file path not found. Please re-upload the file.")

    df = session.duck_conn.execute("SELECT * FROM data").df()
    dirty_idx = _identify_dirty_rows(df)

    if len(dirty_idx) == 0:
        return SearchResult(reply="The dataset looks clean — no dirty rows were detected!")

    dirty_df = df.loc[dirty_idx]
    clean_df = df.drop(dirty_idx)

    reference_df = clean_df.dropna().head(5)
    if reference_df.empty:
        reference_df = df.head(5)

    schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    reference_json = reference_df.to_json(orient="records", date_format="iso")

    from config import CLEANSE_PROMPT

    corrected_rows: list[dict] = []
    total_batches = (len(dirty_df) + MAX_DIRTY_BATCH - 1) // MAX_DIRTY_BATCH

    for batch_num, start in enumerate(range(0, len(dirty_df), MAX_DIRTY_BATCH), 1):
        batch = dirty_df.iloc[start: start + MAX_DIRTY_BATCH].copy()
        batch.insert(0, "_row_index", batch.index.tolist())
        batch_json = batch.to_json(orient="records", date_format="iso")

        print(f"[cleanse] batch {batch_num}/{total_batches} — {len(batch)} dirty rows")

        prompt = CLEANSE_PROMPT.format(
            schema=schema_str,
            reference_rows=reference_json,
            dirty_rows=batch_json,
        )
        raw = strip_fences(get_response_from_llm(prompt))

        try:
            result = json.loads(raw)
            if isinstance(result, list):
                corrected_rows.extend(result)
            else:
                print(f"[cleanse] batch {batch_num} — unexpected response structure, skipping")
        except json.JSONDecodeError as exc:
            print(f"[cleanse] batch {batch_num} — invalid JSON: {exc}")

    result_df = df.copy()
    applied = 0
    for row in corrected_rows:
        idx = row.pop("_row_index", None)
        if idx is None or idx not in result_df.index:
            continue
        for col, val in row.items():
            if col in result_df.columns:
                result_df.at[idx, col] = val
        applied += 1

    base, ext = os.path.splitext(session.current_filepath)
    cleansed_path = f"{base}_cleansed{ext}"
    cleansed_filename = os.path.basename(cleansed_path)

    if ext.lower() == ".csv":
        result_df.to_csv(cleansed_path, index=False)
    else:
        result_df.to_excel(cleansed_path, index=False)

    print(f"[cleanse] saved → {cleansed_path}")

    summary = (
        f"Data cleansing complete!\n\n"
        f"- **{len(dirty_idx)}** dirty rows detected out of **{len(df)}** total\n"
        f"- **{applied}** rows corrected by the AI\n"
        f"- Cleansed file: `{cleansed_filename}`\n\n"
        f"Use the download button below to get the cleansed file."
    )
    return SearchResult(
        reply=summary,
        download_url=f"/api/download/{cleansed_filename}",
    )
