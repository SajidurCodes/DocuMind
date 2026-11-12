from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    text: str
    page_number: int | None = Field(default=None, ge=1)