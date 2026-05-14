from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class SearchResult:
    reply: str
    vega_spec: dict | None = field(default=None)
    insights: list = field(default_factory=list)
    download_url: str | None = field(default=None)


class SmartParserQuestion(BaseModel):
    id: str
    label: str
    placeholder: str
    required: bool


class SmartParserAnswer(BaseModel):
    id: str
    answer: str


class SmartParserAnswersRequest(BaseModel):
    session_id: str | None = None
    answers: list[SmartParserAnswer]


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