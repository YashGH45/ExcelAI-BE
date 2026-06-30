import dataclasses
import shutil

import chromadb

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from session import SessionManager
from config import UPLOAD_DIR
from dto_models import (
    ChatRequest, ChatResponse, UploadResponse,
    SmartParserQuestion, SmartParserAnswersRequest, SearchResult,
)
from search import search
from pdf_handler import load_pdf_index
from metadata import store_metadata_in_vectordb, generate_rich_metadata, generate_smart_parser_questions, generate_schema_summary
from file_loader import load_dataframe, load_pdf_pages, load_into_duckdb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_methods=["*"],
    allow_headers=["*"],
)

_chroma_client = chromadb.PersistentClient(path="./vector_db")
session_manager = SessionManager(_chroma_client)

# Fallback session id used when the client doesn't supply one
_ANON = "__anon__"


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    sid = body.session_id or _ANON
    session = session_manager.get(sid)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a file first.",
        )
    result: SearchResult = search(body.message, session)
    return ChatResponse(
        reply=result.reply,
        vega_spec=result.vega_spec,
        insights=result.insights,
        download_url=result.download_url,
        sources=[dataclasses.asdict(s) for s in result.sources],
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
async def upload_file(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    smart_parser: str = Form(default="false"),
):
    smart_parser_mode = smart_parser.strip().lower() == "true"

    filename = file.filename
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    sid = session_id or _ANON

    try:
        session = session_manager.reset(sid)

        # ── PDF file ──────────────────────────────────────────────────────────
        if filename.lower().endswith(".pdf"):
            pages = load_pdf_pages(str(file_path))
            if not pages:
                raise ValueError(
                    "No extractable text found in this PDF. "
                    "Scanned/image-only PDFs are not supported."
                )
            load_pdf_index(pages, session)
            session.current_mode = "pdf"
            session.current_filepath = str(file_path)
            return UploadResponse(
                filename=filename,
                rows=len(pages),
                col_count=0,
                columns=[],
                file_type="pdf",
            )


        # ── Excel / CSV file ──────────────────────────────────────────────────
        df, sheet_name = load_dataframe(str(file_path))
        load_into_duckdb(df, session)
        session.sheet_name = sheet_name
        session.smart_parser_df = df

        if not smart_parser_mode:
            # Generate metadata immediately when smart parser is not used.
            try:
                metadata = generate_rich_metadata(df)
                store_metadata_in_vectordb(metadata, session)
                session.dataset_summary = generate_schema_summary(metadata)
            except Exception as meta_err:
                print(f"[metadata] generation failed (non-fatal): {meta_err}")
        # smart_parser_mode: metadata generation is deferred until the client
        # POSTs answers to POST /api/smart-parser/answers.

        session.current_filepath = str(file_path)
        session.current_mode = "excel"

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process file: {e}")

    return UploadResponse(
        filename=filename,
        rows=len(df),
        col_count=len(df.columns),
        columns=list(df.columns),
        file_type="excel",
    )


# ── Smart-parser Q&A endpoints ───────────────────────────────────────────────

@app.get("/api/smart-parser/questions", response_model=list[SmartParserQuestion])
async def smart_parser_questions(session_id: str | None = None):
    sid = session_id or _ANON
    session = session_manager.get(sid)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload a file via POST /api/upload first.",
        )
    df = session.smart_parser_df
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="No file loaded. Upload a file via POST /api/upload first.",
        )
    try:
        questions = generate_smart_parser_questions(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {e}")
    return questions


@app.post("/api/smart-parser/answers")
async def smart_parser_submit_answers(body: SmartParserAnswersRequest):
    sid = body.session_id or _ANON
    session = session_manager.get(sid)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload a file via POST /api/upload first.",
        )
    df = session.smart_parser_df
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="No file loaded. Upload a file via POST /api/upload first.",
        )

    answers = [a.model_dump() for a in body.answers]
    try:
        metadata = generate_rich_metadata(df, answers=answers)
        store_metadata_in_vectordb(metadata, session)
        session.dataset_summary = generate_schema_summary(metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate metadata: {e}")

    return {"status": "ok"}
