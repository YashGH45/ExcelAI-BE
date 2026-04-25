import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ai_client import search, delete, store_one, load_into_duckdb
from config import UPLOAD_DIR
from dto_models import ChatRequest, ChatResponse, UploadResponse
from services import load_dataframe, convert_rows_to_sentences, generate_schema_and_template

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_methods=["*"],
    allow_headers=["*"],
)


DEV_MODE = True
DEV_FILE = "Hotels_data.xlsx"


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    results = search(body.message)
    reply = "\n\n".join(results)
    return ChatResponse(reply=reply)


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        delete()

        df = load_dataframe(str(file_path))

        # Register dataframe in DuckDB for the text-to-SQL path
        load_into_duckdb(df)

        result = generate_schema_and_template(df)
        summary = result["summary"]
        template = result["template"]

        store_one(summary, 'file_summary')
        convert_rows_to_sentences(df, template)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process file: {e}")

    return UploadResponse(
        filename=filename,
        rows=len(df),
        col_count=len(df.columns),
        columns=list(df.columns),
    )