import time
from fastapi.testclient import TestClient

from src.app import api


class DummyRAG:
    def ask(self, question: str, top_k: int = 3):
        return {
            "question": question,
            "answer": "Dummy answer",
            "confidence": 0.5,
            "sources": [],
            "num_sources": 0,
            "status": "success"
        }


def test_ask_happy_path(monkeypatch):
    monkeypatch.setenv("LEGALADVISOR_SKIP_RAG_INIT", "1")
    monkeypatch.setenv("LEGALADVISOR_LOG_QUESTIONS", "0")

    # Stub RAG
    monkeypatch.setattr(api, "rag_system", DummyRAG())
    monkeypatch.setattr(api, "rag_last_error", None)

    client = TestClient(api.app)

    resp = client.post("/ask", json={"question": "Xin chào", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["answer"] == "Dummy answer"


def test_question_too_long_rejected(monkeypatch):
    monkeypatch.setenv("LEGALADVISOR_SKIP_RAG_INIT", "1")

    client = TestClient(api.app)
    long_q = "a" * 2000  # vượt quá 1024
    resp = client.post("/ask", json={"question": long_q, "top_k": 3})
    assert resp.status_code == 422  # Pydantic validation error


def test_top_k_out_of_range_rejected(monkeypatch):
    monkeypatch.setenv("LEGALADVISOR_SKIP_RAG_INIT", "1")

    client = TestClient(api.app)
    # top_k = 0 -> invalid
    resp = client.post("/ask", json={"question": "hi", "top_k": 0})
    assert resp.status_code == 422
    # top_k = 100 -> invalid
    resp = client.post("/ask", json={"question": "hi", "top_k": 100})
    assert resp.status_code == 422
