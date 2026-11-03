import pytest
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_task4_health(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_task4_missing_api_key(client):
    response = client.post("/sessions")
    assert response.status_code in (401, 403, 422)
def test_task4_invalid_api_key(client):
    response = client.post("/sessions", headers={"x-api-key": "incorrect-key"})
    assert response.status_code in (401, 403, 422)
def test_task4_create_session(client):
    from app.config import settings
    response = client.post("/sessions", headers={"x-api-key": settings.API_KEY})
    assert response.status_code in (200, 201)

def test_task4_unknown_tool(client):
    from app.config import settings
    response = client.post(
        "/sessions/unknown/messages",
        headers={"x-api-key": settings.API_KEY},
    )
    assert response.status_code == 404