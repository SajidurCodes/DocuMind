from fastapi import FastAPI

app = FastAPI(
    title="DocuMind API",
    description="A document question-answering API.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}