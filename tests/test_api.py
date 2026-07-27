import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "qdrant_status" in data
    assert "embedding_model" in data

def test_query_endpoint():
    payload = {
        "query": "What is the remote work monthly stipend?",
        "top_k": 3,
        "metadata_filter": {
            "category": "policy"
        }
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "performance" in data
    assert "token_usage" in data
