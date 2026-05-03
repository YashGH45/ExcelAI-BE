from pydantic import BaseModel


class SmartParserQuestion(BaseModel):
    id: str
    label: str
    placeholder: str
    required: bool


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    vega_spec: dict | None = None
    insights: list[str] = []
    download_url: str | None = None


class UploadResponse(BaseModel):
    filename: str
    rows: int
    col_count: int
    columns: list[str]
    file_type: str = "excel"   # "excel" | "pdf"