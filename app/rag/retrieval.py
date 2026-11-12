import re

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_index: int
    text: str
    score: int


def tokenize(text: str) -> set[str]:
    """Return unique lowercase words and numbers."""
    return set(re.findall(r"\w+", text.casefold()))


def retrieve_chunks(
    question: str,
    chunks: list[str],
    top_k: int = 3,
) -> list[RetrievedChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    question_words = tokenize(question)
    matches = []

    for index, chunk in enumerate(chunks):
        chunk_words = tokenize(chunk)
        score = len(question_words & chunk_words)

        if score > 0:
            matches.append(
                RetrievedChunk(
                    chunk_index=index,
                    text=chunk,
                    score=score,
                )
            )

    matches.sort(key=lambda match: (-match.score, match.chunk_index))
    return matches[:top_k]