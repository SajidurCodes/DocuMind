from io import BytesIO

from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from app.rag.chunking import split_text
from app.rag.models import DocumentChunk


class PDFPage(BaseModel):
    page_number: int
    text: str


def extract_pdf_pages(content: bytes) -> list[PDFPage]:
    """Extract text while preserving original PDF page numbers."""
    if not content:
        raise ValueError("The PDF file is empty.")

    pages = []

    try:
        with BytesIO(content) as stream:
            reader = PdfReader(stream)

            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported yet.")

            if len(reader.pages) > 50:
                raise ValueError("PDFs must contain no more than 50 pages.")

            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()

                # Skip pages without text, retaining original page numbers.
                if text:
                    pages.append(
                        PDFPage(page_number=number, text=text)
                    )

    except PdfReadError as exc:
        raise ValueError(
            "The PDF could not be read. It may be damaged or invalid."
        ) from exc

    if not pages:
        raise ValueError(
            "No extractable text was found. "
            "Scanned PDFs may require OCR."
        )

    return pages


def chunk_pdf_pages(
    pages: list[PDFPage],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[DocumentChunk]:
    chunks = []

    for page in pages:
        page_chunks = split_text(
            text=page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for text in page_chunks:
            chunks.append(
                DocumentChunk(
                    text=text,
                    page_number=page.page_number,
                )
            )

    return chunks