import duckdb
import pandas as pd
import chromadb
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi


@dataclass
class Session:
    session_id: str
    # ── Data state ────────────────────────────────────────────────────────────
    duck_conn: duckdb.DuckDBPyConnection | None = None
    current_filepath: str | None = None
    rich_metadata: dict | None = None
    smart_parser_df: pd.DataFrame | None = None
    dataset_summary: str | None = None
    current_mode: str = "excel"
    sheet_name: str | None = None
    # ── Excel conversation memory ─────────────────────────────────────────────
    recent_turns: deque = field(default_factory=lambda: deque(maxlen=5))
    conversation_summary: str = ""
    # ── PDF state ─────────────────────────────────────────────────────────────
    pdf_pages: list[dict] | None = None
    pdf_bm25: BM25Okapi | None = None
    recent_pdf_turns: deque = field(default_factory=lambda: deque(maxlen=5))
    pdf_conversation_summary: str = ""
    # ── ChromaDB collection handles (not serialised — re-acquired by name) ────
    metadata_collection: Any = field(default=None, repr=False)
    conv_collection: Any = field(default=None, repr=False)
    meta_collection_name: str = ""
    conv_collection_name: str = ""


class SessionManager:
    def __init__(self, chroma_client: chromadb.PersistentClient):
        self._store: dict[str, Session] = {}
        self._chroma = chroma_client

    def create(self, session_id: str) -> Session:
        """Create fresh per-session ChromaDB collections and store a new Session."""
        meta_name = f"rich_metadata_{session_id}"
        conv_name = f"conv_memory_{session_id}"
        for name in (meta_name, conv_name):
            try:
                self._chroma.delete_collection(name)
            except Exception:
                pass
        session = Session(
            session_id=session_id,
            metadata_collection=self._chroma.get_or_create_collection(meta_name),
            conv_collection=self._chroma.get_or_create_collection(conv_name),
            meta_collection_name=meta_name,
            conv_collection_name=conv_name,
        )
        self._store[session_id] = session
        print(f"[session] created {session_id[:8]}…")
        return session

    def get(self, session_id: str | None) -> Session | None:
        if session_id is None:
            return None
        return self._store.get(session_id)

    def reset(self, session_id: str) -> Session:
        """Wipe the existing session state (if any) and return a fresh one.
        Absorbs the old delete() + prepare_session() logic."""
        old = self._store.pop(session_id, None)
        if old:
            for name in (old.meta_collection_name, old.conv_collection_name):
                if name:
                    try:
                        self._chroma.delete_collection(name)
                    except Exception:
                        pass
        return self.create(session_id)