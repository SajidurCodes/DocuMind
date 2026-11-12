import re

from pydantic import BaseModel

from app.rag.models import DocumentChunk


class RetrievedChunk(BaseModel):
    chunk_index: int
    text: str
    score: int
    page_number: int | None = None


def tokenize(text: str) -> set[str]:
    """Return unique lowercase words and numbers."""
    return set(re.findall(r"\w+", text.casefold()))


def retrieve_chunks(
    question: str,
    chunks: list[str] | list[DocumentChunk],
    top_k: int = 3,
) -> list[RetrievedChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    question_words = tokenize(question)
    matches = []

    for index, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            text = chunk
            page_number = None
        else:
            text = chunk.text
            page_number = chunk.page_number

        score = len(question_words & tokenize(text))

        if score > 0:
            matches.append(
                RetrievedChunk(
                    chunk_index=index,
                    text=text,
                    score=score,
                    page_number=page_number,
                )
            )

    matches.sort(key=lambda match: (-match.score, match.chunk_index))
    return matches[:top_k]