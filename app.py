import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ai_client import (
    search, delete, store_one, load_into_duckdb,
    set_dataset_summary, load_pdf_index, set_mode, set_current_filepath,
    store_metadata_in_vectordb, SearchResult,
)
from config import UPLOAD_DIR
from dto_models import ChatRequest, ChatResponse, UploadResponse
from services import (
    load_dataframe, convert_rows_to_sentences,
    generate_schema_and_template, generate_rich_metadata, load_pdf_pages,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_methods=["*"],
    allow_headers=["*"],
)

USE_VECTOR_SEARCH = False

DEV_MODE = True
DEV_FILE = "Hotels_data.xlsx"


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    result: SearchResult = search(body.message)
    return ChatResponse(
        reply=result.reply,
        vega_spec=result.vega_spec,
        insights=result.insights,
        download_url=result.download_url,
    )


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        delete()

        if filename.lower().endswith(".pdf"):
            # ── PDF path: VectorLess BM25 PageIndex ──────────────────────────
            pages = load_pdf_pages(str(file_path))
            if not pages:
                raise ValueError(
                    "No extractable text found in this PDF. "
                    "Scanned/image-only PDFs are not supported."
                )
            load_pdf_index(pages)
            set_mode("pdf")
            return UploadResponse(
                filename=filename,
                rows=len(pages),
                col_count=0,
                columns=[],
                file_type="pdf",
            )

        # ── Excel / CSV path: Text-to-SQL ─────────────────────────────────────
        df = load_dataframe(str(file_path))

        if USE_VECTOR_SEARCH:
            result = generate_schema_and_template(df)
            summary = result["summary"]
            template = result["template"]

            set_dataset_summary(summary)
            store_one(summary, 'file_summary')
            convert_rows_to_sentences(df, template)

        else:
            load_into_duckdb(df)

            try:
                metadata = generate_rich_metadata(df)
                store_metadata_in_vectordb(metadata)
            except Exception as meta_err:
                print(f"[metadata] rich metadata generation failed (non-fatal): {meta_err}")

        set_current_filepath(str(file_path))
        set_mode("excel")

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process file: {e}")

    return UploadResponse(
        filename=filename,
        rows=len(df),
        col_count=len(df.columns),
        columns=list(df.columns),
        file_type="excel",
    )