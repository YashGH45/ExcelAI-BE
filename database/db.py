import sqlite3
from pathlib import Path
import uuid

# Database file location
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chat_history.db"


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def _add_column_if_missing(cursor, table: str, column: str, definition: str):
    cursor.execute(f"PRAGMA table_info({table})")
    existing = [row["name"] for row in cursor.fetchall()]
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def initialize_database():

    connection = get_db_connection()
    cursor = connection.cursor()

    # Table 1: Chats
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 2: Messages
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (chat_id)
                REFERENCES chats(id)
                ON DELETE CASCADE
        )
    """)

    # Index for faster message retrieval by chat
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_chat_id
        ON messages(chat_id)
    """)

    # add file columns to chats (safe to run on an existing database)
    _add_column_if_missing(cursor, "chats", "file_path", "TEXT")
    _add_column_if_missing(cursor, "chats", "file_type", "TEXT")

    connection.commit()
    connection.close()

    
def create_chat(title: str):
    # Generate a unique ID for this conversation
    chat_id = f"chat_{uuid.uuid4().hex}"

    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO chats (id, title)
            VALUES (?, ?)
            """,
            (chat_id, title)
        )

        connection.commit()

        return {
            "id": chat_id,
            "title": title
        }

    finally:
        connection.close()



def get_all_chats():

    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT *
            FROM chats
            ORDER BY updated_at DESC
        """)

        chats = cursor.fetchall()

        return [dict(chat) for chat in chats]

    finally:
        connection.close()


def save_message(chat_id: str, role: str, content: str):
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO messages (chat_id, role, content)
                VALUES (?, ?, ?)
                """,
                (chat_id, role, content)
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Cannot save message: chat '{chat_id}' does not exist")

        cursor.execute(
            """
            UPDATE chats
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (chat_id,)
        )

        connection.commit()

    finally:
        connection.close()


def get_chat_messages(chat_id: str):
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM messages
            WHERE chat_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (chat_id,)
        )

        messages = cursor.fetchall()

        return [dict(message) for message in messages]

    finally:
        connection.close()


def delete_chat(chat_id: str) -> bool:
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM chats WHERE id = ?",
            (chat_id,)
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:
        connection.close()


def get_recent_messages(chat_id: str, limit: int = 10):
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit)
        )

        rows = cursor.fetchall()

        # We fetched newest-first, so flip it back to oldest-first
        return [dict(row) for row in reversed(rows)]

    finally:
        connection.close()


def rename_chat(chat_id: str, new_title: str) -> bool:
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE chats
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_title, chat_id)
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:
        connection.close()


def ensure_chat(chat_id: str, title: str = "New chat") -> bool:
    """Create the chat if it doesn't exist yet. Returns True if it was created."""
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            "INSERT OR IGNORE INTO chats (id, title) VALUES (?, ?)",
            (chat_id, title)
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:
        connection.close()
        

def chat_exists(chat_id: str) -> bool:
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT 1 FROM chats WHERE id = ? LIMIT 1",
            (chat_id,)
        )

        return cursor.fetchone() is not None

    finally:
        connection.close()


def set_chat_file(chat_id: str, file_path: str, file_type: str) -> None:
    connection = get_db_connection()
    try:
        connection.execute(
            "UPDATE chats SET file_path = ?, file_type = ? WHERE id = ?",
            (file_path, file_type, chat_id),
        )
        connection.commit()
    finally:
        connection.close()


def get_chat(chat_id: str):
    connection = get_db_connection()
    try:
        row = connection.execute(
            "SELECT * FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()