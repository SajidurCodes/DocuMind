import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.storage as storage
from app.rag.chunking import split_text
from app.rag.models import DocumentChunk
from app.rag.retrieval import retrieve_chunks


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Use a separate temporary database for every test.
    monkeypatch.setattr(
        storage,
        "DATABASE_PATH",
        tmp_path / "test.db",
    )

    # Fail immediately if a test accidentally calls the real Groq client.
    def block_groq(*args, **kwargs):
        raise AssertionError("Tests must not call the real Groq API.")

    monkeypatch.setattr(main, "Groq", block_groq)

    with TestClient(main.app) as test_client:
        yield test_client


def test_chunk_overlap():
    chunks = split_text(
        "ABCDEFGHIJKLMN",
        chunk_size=6,
        overlap=2,
    )

    assert chunks == ["ABCDEF", "EFGHIJ", "IJKLMN"]


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("Example", chunk_size=6, overlap=6)


def test_retrieval_preserves_page():
    chunks = [
        DocumentChunk(text="Annual leave is 18 days.", page_number=1),
        DocumentChunk(text="Expense claims are due in 14 days.", page_number=2),
    ]

    matches = retrieve_chunks("Expense claims", chunks, top_k=1)

    assert len(matches) == 1
    assert matches[0].chunk_index == 1
    assert matches[0].page_number == 2


def test_txt_upload_and_read(client):
    response = client.post(
        "/documents",
        files={
            "file": (
                "handbook.txt",
                b"Employees receive 18 days of annual leave.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    saved = response.json()

    loaded = client.get(f"/documents/{saved['document_id']}")

    assert loaded.status_code == 200
    assert loaded.json() == saved
    assert saved["chunks"][0]["page_number"] is None


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("empty.txt", b"", 422),
        ("spaces.txt", b"   ", 422),
        ("script.py", b"print('hello')", 415),
        ("invalid.txt", b"\xff\xfe", 422),
        ("large.txt", b"x" * (1024 * 1024 + 1), 413),
    ],
)
def test_invalid_uploads(client, filename, content, expected_status):
    response = client.post(
        "/documents",
        files={"file": (filename, content)},
    )

    assert response.status_code == expected_status


def test_missing_document(client):
    response = client.get("/documents/does-not-exist")

    assert response.status_code == 404


def test_blank_question(client):
    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422


def test_answer_preserves_source_page(client, monkeypatch):
    document = storage.save_document(
        "sample.pdf",
        [
            DocumentChunk(
                text="Expense claims must be submitted within 14 days.",
                page_number=2,
            )
        ],
    )

    def fake_answer(question, system_prompt):
        # Confirm the document evidence is actually sent to the model.
        assert "Expense claims must be submitted within 14 days." in question
        assert "using only the supplied document excerpts" in system_prompt
        return main.ChatResponse(answer="Within 14 days [1].")

    monkeypatch.setattr(main, "generate_answer", fake_answer)

    response = client.post(
        f"/documents/{document.document_id}/ask",
        json={"question": "When must expense claims be submitted?"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "Within 14 days [1]."
    assert result["sources"][0]["page_number"] == 2
    assert result["sources"][0]["citation_id"] == 1


def test_no_matches_skips_model(client):
    document = storage.save_document(
        "handbook.txt",
        [DocumentChunk(text="Annual leave is 18 days.")],
    )

    response = client.post(
        f"/documents/{document.document_id}/ask",
        json={"question": "astronomy galaxies"},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["answer"] == (
        "I couldn't find relevant information in this document."
    )