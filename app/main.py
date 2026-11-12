import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import APIConnectionError, APIStatusError, Groq
from pydantic import BaseModel, StringConstraints


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