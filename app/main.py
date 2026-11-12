import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from groq import APIConnectionError, APIStatusError, Groq
from pydantic import BaseModel, StringConstraints
from app.rag.chunking import split_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="DocuMind API",
    description="A document question-answering API.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=2000,
        ),
    ]


class ChatResponse(BaseModel):
    answer: str


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest) -> ChatResponse:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="The AI service is not configured.",
        )

    try:
        with Groq(
            api_key=api_key,
            timeout=30.0,
            max_retries=0,
        ) as client:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": "Give clear, concise answers.",
                    },
                    {
                        "role": "user",
                        "content": request.question,
                    },
                ],
                reasoning_effort="low",
                max_completion_tokens=1024,
            )
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach the AI service. Try again later.",
        ) from exc
    except APIStatusError as exc:
        if exc.status_code == 429:
            raise HTTPException(
                status_code=503,
                detail="The AI service usage limit was reached. Try later.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="The AI service could not complete the request.",
        ) from exc

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise HTTPException(
            status_code=502,
            detail="The AI service returned an empty answer.",
        )

    return ChatResponse(answer=answer.strip())



class DocumentPreview(BaseModel):
    filename: str
    character_count: int
    preview: str
    chunk_count: int
    chunks: list[str]


@app.post(
    "/documents/preview",
    response_model=DocumentPreview,
    tags=["Documents"],
)
async def preview_document(file: UploadFile) -> DocumentPreview:
    max_bytes = 1024 * 1024  # 1 MiB

    try:
        filename = file.filename or ""

        if Path(filename).suffix.lower() != ".txt":
            raise HTTPException(
                status_code=415,
                detail="Only .txt files are supported for now.",
            )

        content = await file.read(max_bytes + 1)

        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="The file must be no larger than 1 MiB.",
            )

        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail="Save the text file using UTF-8 encoding.",
            ) from exc

        if not text:
            raise HTTPException(
                status_code=422,
                detail="The file contains no text.",
            )

        chunks = split_text(text)

        return DocumentPreview(
            filename=filename,
            character_count=len(text),
            preview=text[:500],
            chunk_count=len(chunks),
            chunks=chunks,
        )
    finally:
        await file.close()