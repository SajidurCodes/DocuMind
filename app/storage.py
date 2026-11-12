import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "local" / "documind.db"


class StoredDocument(BaseModel):
    document_id: str
    filename: str
    chunks: list[str]


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    chunks_json TEXT NOT NULL
                )
                """
            )


def save_document(filename: str, chunks: list[str]) -> StoredDocument:
    if not chunks or any(not chunk.strip() for chunk in chunks):
        raise ValueError("A document must contain nonempty chunks.")

    document = StoredDocument(
        document_id=uuid4().hex,
        filename=filename,
        chunks=chunks,
    )

    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO documents (document_id, filename, chunks_json)
                VALUES (?, ?, ?)
                """,
                (
                    document.document_id,
                    document.filename,
                    json.dumps(document.chunks),
                ),
            )

    return document


def get_document(document_id: str) -> StoredDocument | None:
    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        row = connection.execute(
            """
            SELECT document_id, filename, chunks_json
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

    if row is None:
        return None

    return StoredDocument(
        document_id=row[0],
        filename=row[1],
        chunks=json.loads(row[2]),
    )