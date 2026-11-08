import dataclasses
import shutil
import uuid
import os

import chromadb

from fastapi import FastAPI, Form, Path, UploadFile, File, HTTPException
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

from contextlib import asynccontextmanager
from database.db import (
    initialize_database,
    ensure_chat,
    save_message,
    get_all_chats,
    get_chat_messages,
    chat_exists,
    delete_chat,
    get_chat,
    set_chat_file
)

initialize_database()

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_methods=["*"],
    allow_headers=["*"],
)

_chroma_client = chromadb.PersistentClient(path="./vector_db")
session_manager = SessionManager(_chroma_client)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    sid = body.session_id
    session = session_manager.get(sid)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a file first.",
        )

    ensure_chat(sid)      
    save_message(sid, "user", body.message)          

    result: SearchResult = search(body.message, session)

    save_message(sid, "assistant", result.reply)     

    return ChatResponse(
        reply=result.reply,
        vega_spec=result.vega_spec,
        insights=result.insights,
        download_url=result.download_url,
        sources=[dataclasses.asdict(s) for s in result.sources],
        evidence=[dataclasses.asdict(e) for e in result.evidence],
    )


@app.get("/api/chats")
async def list_chats():
    return get_all_chats()


@app.get("/api/chats/{chat_id}/messages")
async def list_chat_messages(chat_id: str):
    if not chat_exists(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return get_chat_messages(chat_id)


@app.delete("/api/chats/{chat_id}")
async def remove_chat(chat_id: str):
    if not delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    session_manager.delete(chat_id)  
    return {"deleted": chat_id}


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

def _restore_file_into_session(session, file_path: str) -> None:
    if file_path.lower().endswith(".pdf"):
        pages = load_pdf_pages(file_path)
        if not pages:
            raise ValueError("No extractable text found in this PDF.")
        load_pdf_index(pages, session)
        session.current_mode = "pdf"
        session.current_filepath = file_path
        return

    df, sheet_name = load_dataframe(file_path)
    load_into_duckdb(df, session)
    session.sheet_name = sheet_name
    session.smart_parser_df = df
    try:
        metadata = generate_rich_metadata(df)
        store_metadata_in_vectordb(metadata, session)
        session.dataset_summary = generate_schema_summary(metadata)
    except Exception as meta_err:
        print(f"[metadata] generation failed (non-fatal): {meta_err}")
    session.current_filepath = file_path
    session.current_mode = "excel"


@app.post("/api/chats/{chat_id}/restore")
async def restore_chat(chat_id: str):
    if session_manager.get(chat_id) is not None:
        return {"status": "already_active"}

    chat = get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found.")

    file_path = chat.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original file is no longer available.")

    try:
        session = session_manager.create(chat_id)
        _restore_file_into_session(session, file_path)
    except Exception as e:
        session_manager.delete(chat_id)
        raise HTTPException(status_code=422, detail=f"Failed to restore chat: {e}")

    return {"status": "restored"}

    
@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    smart_parser: str = Form(default="false"),
):
    smart_parser_mode = smart_parser.strip().lower() == "true"

    filename = file.filename
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    sid = str(uuid.uuid4())

    try:
        session = session_manager.create(sid)

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
            ensure_chat(sid, title=filename)
            set_chat_file(sid, str(file_path), "pdf")
            return UploadResponse(
                filename=filename,
                rows=len(pages),
                col_count=0,
                columns=[],
                file_type="pdf",
                session_id=sid,
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
        ensure_chat(sid, title=filename) 
        set_chat_file(sid, str(file_path), "excel") 

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process file: {e}")

    return UploadResponse(
        filename=filename,
        rows=len(df),
        col_count=len(df.columns),
        columns=list(df.columns),
        file_type="excel",
        session_id=sid, 
    )

# ── Smart-parser Q&A endpoints ───────────────────────────────────────────────

@app.get("/api/smart-parser/questions", response_model=list[SmartParserQuestion])
async def smart_parser_questions(session_id: str | None = None):
    sid = session_id 
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
    sid = body.session_id 
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
