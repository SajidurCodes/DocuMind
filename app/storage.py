import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from app.rag.models import DocumentChunk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "local" / "documind.db"


class StoredDocument(BaseModel):
    document_id: str
    filename: str
    chunks: list[DocumentChunk]


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


def save_document(
    filename: str,
    chunks: list[str] | list[DocumentChunk],
) -> StoredDocument:
    normalized_chunks = [
        DocumentChunk(text=chunk) if isinstance(chunk, str) else chunk
        for chunk in chunks
    ]

    if not normalized_chunks or any(
        not chunk.text.strip() for chunk in normalized_chunks
    ):
        raise ValueError("A document must contain nonempty chunks.")

    document = StoredDocument(
        document_id=uuid4().hex,
        filename=filename,
        chunks=normalized_chunks,
    )

    chunks_json = json.dumps(
        [chunk.model_dump() for chunk in document.chunks]
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
                    chunks_json,
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

    stored_chunks = json.loads(row[2])

    # Older records contain strings; newer records contain dictionaries.
    chunks = [
        DocumentChunk(text=chunk)
        if isinstance(chunk, str)
        else DocumentChunk.model_validate(chunk)
        for chunk in stored_chunks
    ]

    return StoredDocument(
        document_id=row[0],
        filename=row[1],
        chunks=chunks,
    )