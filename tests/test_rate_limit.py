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


def test_rate_limit_enforced(monkeypatch):
    monkeypatch.setenv("LEGALADVISOR_SKIP_RAG_INIT", "1")

    monkeypatch.setattr(api, "rag_system", DummyRAG())
    monkeypatch.setattr(api, "rag_last_error", None)
    api._rate_limit_store.clear()

    monkeypatch.setattr(api, "RATE_LIMIT_MAX", 3, raising=False)
    monkeypatch.setattr(api, "RATE_LIMIT_WINDOW", 1, raising=False)

    client = TestClient(api.app)

    for _ in range(api.RATE_LIMIT_MAX):
        response = client.post("/ask", json={"question": "Test", "top_k": 1})
        assert response.status_code == 200

    response = client.post("/ask", json={"question": "Test", "top_k": 1})
    assert response.status_code == 429
    data = response.json()
    assert data.get("message") == "Rate limit exceeded"

    time.sleep(api.RATE_LIMIT_WINDOW)
    response = client.post("/ask", json={"question": "Test", "top_k": 1})
    assert response.status_code == 200
