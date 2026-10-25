import dataclasses
from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class SpreadsheetSource:
    type: str = "spreadsheet"
    file_name: str = ""
    sheet_name: str | None = None
    sql: str = ""
    retrieved_row_count: int = 0
    total_row_count: int = 0
    columns_used: list = field(default_factory=list)
    rows: list = field(default_factory=list)


@dataclass
class PDFSource:
    type: str = "pdf"
    file_name: str = ""
    page_number: int = 0
    section_heading: str | None = None
    text: str = ""
    score: float = 0.0
    highlight_spans: list = field(default_factory=list)


@dataclass
class PDFEvidence:
    chunk_id: str = ""
    page: int = 0
    heading: str | None = None
    paragraph_ids: list = field(default_factory=list)
    bounding_boxes: list = field(default_factory=list)
    character_offsets: dict = field(default_factory=dict)
    similarity_score: float = 0.0
    evidence_text: str | None = None


@dataclass
class SearchResult:
    reply: str
    vega_spec: dict | None = field(default=None)
    insights: list = field(default_factory=list)
    download_url: str | None = field(default=None)
    sources: list = field(default_factory=list)
    evidence: list = field(default_factory=list)


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
    sources: list = []
    evidence: list = []


class UploadResponse(BaseModel):
    filename: str
    rows: int
    col_count: int
    columns: list[str]
    file_type: str = "excel"   # "excel" | "pdf"
