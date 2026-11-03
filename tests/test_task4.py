import pytest
from fastapi.testclient import TestClient


from app.main import app
from app.config import settings


from app.task2 import (
    ProviderError,
    should_retry,
    retrying_call,
    demo_retry,
    backoff_explanation,
)




@pytest.fixture
def client():
    return TestClient(app)




# -------------------------------
# FastAPI Task 4 tests
# -------------------------------


def test_task4_health(client):
    response = client.get("/health")
    assert response.status_code == 200




def test_task4_missing_api_key(client):
    response = client.post("/sessions")
    assert response.status_code == 401




def test_task4_invalid_api_key(client):
    response = client.post(
        "/sessions",
        headers={"x-api-key": "incorrect-key"},
    )
    assert response.status_code == 403




def test_task4_create_session(client):
    response = client.post(
        "/sessions",
        headers={"x-api-key": settings.API_KEY},
    )
    assert response.status_code in (200, 201)




def test_task4_unknown_session(client):
    response = client.get(
        "/sessions/unknown/messages",
        headers={"x-api-key": settings.API_KEY},
    )
    assert response.status_code == 404




# -------------------------------
# Retry tests
# These functions belong to task2.py
# -------------------------------


def test_should_retry_429():
    assert should_retry(ProviderError(429)) is True




def test_should_retry_500():
    assert should_retry(ProviderError(500)) is True




def test_should_retry_503():
    assert should_retry(ProviderError(503)) is True




def test_should_not_retry_401():
    assert should_retry(ProviderError(401)) is False




def test_should_not_retry_403():
    assert should_retry(ProviderError(403)) is False




def test_should_not_retry_regular_exception():
    assert should_retry(ValueError("Invalid input")) is False




def test_retrying_call_success():
    result = retrying_call(lambda: "Success")
    assert result == "Success"




def test_retrying_call_returns_number():
    result = retrying_call(lambda: 123)
    assert result == 123




def test_demo_retry():
    result = demo_retry()


    assert result["result"] == "Success after retries"
    assert result["attempts"] == 3




def test_backoff_explanation():
    explanation = backoff_explanation()


    assert len(explanation) >= 4
    assert any("jitter" in item.lower() for item in explanation)
    assert any("401" in item for item in explanation)




def test_provider_error_message():
    error = ProviderError(503, "Temporary error")


    assert error.status_code == 503
    assert str(error) == "Temporary error"