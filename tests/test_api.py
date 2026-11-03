import asyncio
import json


import pytest
from fastapi.testclient import TestClient


import app.task1 as task1
import app.task2 as task2
import app.task3 as task3
import app.llm as llm


from app.main import app, SESSIONS, get_llm, get_store
from app.config import settings




def fake_llm(prompt):
    return {
        "answer": "This is a mocked answer",
        "model": "test-model",
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }




@pytest.fixture(autouse=True)
def setup_dependencies():
    SESSIONS.clear()
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_store] = lambda: SESSIONS
    yield
    app.dependency_overrides.clear()




@pytest.fixture
def client():
    return TestClient(app)




def auth():
    return {"x-api-key": settings.API_KEY}




# ---------------- API tests ----------------


def test_health(client):
    assert client.get("/health").status_code == 200




def test_missing_key(client):
    assert client.post("/sessions").status_code == 401




def test_invalid_key(client):
    assert client.post(
        "/sessions", headers={"x-api-key": "wrong"}
    ).status_code == 403




def test_create_session(client):
    response = client.post("/sessions", headers=auth())
    assert response.status_code == 200
    assert "session_id" in response.json()




def test_chat_and_history(client):
    sid = client.post("/sessions", headers=auth()).json()["session_id"]


    response = client.post(
        f"/sessions/{sid}/chat",
        headers=auth(),
        json={"message": "Hello"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "This is a mocked answer"


    history = client.get(
        f"/sessions/{sid}/messages", headers=auth()
    )
    assert len(history.json()["messages"]) == 2




def test_empty_message(client):
    sid = client.post("/sessions", headers=auth()).json()["session_id"]
    response = client.post(
        f"/sessions/{sid}/chat",
        headers=auth(),
        json={"message": ""},
    )
    assert response.status_code == 422




def test_message_too_long(client):
    sid = client.post("/sessions", headers=auth()).json()["session_id"]
    response = client.post(
        f"/sessions/{sid}/chat",
        headers=auth(),
        json={"message": "x" * 2001},
    )
    assert response.status_code == 422




def test_unknown_session_chat(client):
    response = client.post(
        "/sessions/unknown/chat",
        headers=auth(),
        json={"message": "Hello"},
    )
    assert response.status_code == 404




def test_unknown_session_history(client):
    response = client.get(
        "/sessions/unknown/messages", headers=auth()
    )
    assert response.status_code == 404




def test_llm_failure_returns_502(client):
    def failing_llm(prompt):
        raise RuntimeError("simulated failure")


    app.dependency_overrides[get_llm] = lambda: failing_llm
    sid = client.post("/sessions", headers=auth()).json()["session_id"]


    response = client.post(
        f"/sessions/{sid}/chat",
        headers=auth(),
        json={"message": "Hello"},
    )
    assert response.status_code == 502




# ---------------- Task 1 tests ----------------


def test_count_tokens():
    messages = [{"role": "user", "content": "Hello world"}]
    assert task1.count_tokens(messages) > 0




def test_usage_logging_and_reading(monkeypatch, tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    monkeypatch.setattr(task1, "USAGE_FILE", usage_file)


    task1.log_usage({
        "date": task1.date.today().isoformat(),
        "total_tokens": 12,
        "cost_usd": 0.01,
    })


    result = task1.get_today_usage()
    assert result["total_tokens"] == 12
    assert result["cost_usd"] == 0.01




def test_usage_ignores_invalid_json(monkeypatch, tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    usage_file.write_text("not-json\n", encoding="utf-8")
    monkeypatch.setattr(task1, "USAGE_FILE", usage_file)


    assert task1.get_today_usage()["total_tokens"] == 0




def test_chat_endpoint(client, monkeypatch, tmp_path):
    monkeypatch.setattr(task1, "USAGE_FILE", tmp_path / "usage.jsonl")
    monkeypatch.setattr(task1, "ask_openai", fake_llm)


    response = client.post(
        "/chat", json={"message": "Hello"}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "This is a mocked answer"




def test_chat_llm_failure(client, monkeypatch):
    def fail(prompt):
        raise RuntimeError("test failure")


    monkeypatch.setattr(task1, "ask_openai", fail)
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 502




def test_chat_input_validation(client):
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422




def test_stream_endpoint(client, monkeypatch):
    class FakeDelta:
        content = "Hello"


    class FakeChoice:
        delta = FakeDelta()


    class FakeChunk:
        choices = [FakeChoice()]


    class FakeCompletions:
        def create(self, **kwargs):
            return [FakeChunk()]


    class FakeChat:
        completions = FakeCompletions()


    class FakeClient:
        chat = FakeChat()


    monkeypatch.setattr(task1, "get_openai_client", lambda: FakeClient())


    response = client.post(
        "/chat/stream", json={"message": "Hello"}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "Hello" in response.text




def test_usage_endpoint(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        task1, "USAGE_FILE", tmp_path / "usage.jsonl"
    )
    response = client.get("/usage")
    assert response.status_code == 200
    assert response.json()["total_tokens"] == 0




# ---------------- Task 2 tests ----------------


def test_retry_success():
    assert task2.retrying_call(lambda: "success") == "success"




def test_retry_only_temporary_errors():
    assert task2.should_retry(task2.ProviderError(429))
    assert task2.should_retry(task2.ProviderError(500))
    assert task2.should_retry(task2.ProviderError(503))
    assert not task2.should_retry(task2.ProviderError(401))
    assert not task2.should_retry(ValueError("bad request"))




def test_backoff_explanation():
    explanation = task2.backoff_explanation()
    assert len(explanation) >= 3
    assert any("401" in item for item in explanation)




def test_async_comparison():
    async def fake_question(question):
        return {"answer": question}


    result = asyncio.run(
        task2.compare_sequential_and_parallel(fake_question)
    )
    assert len(result["sequential_results"]) == 10
    assert len(result["parallel_results"]) == 10




def test_async_partial_failure():
    async def fake_question(question):
        if question == "Question 3":
            raise RuntimeError("simulated")
        return {"answer": question}


    result = asyncio.run(
        task2.compare_sequential_and_parallel(fake_question)
    )
    assert "error" in result["sequential_results"][2]
    assert "error" in result["parallel_results"][2]




# ---------------- Task 3 tests ----------------


def test_order_tool():
    result = task3.get_order_status("1001")
    assert result["status"] == "shipped"




def test_unknown_order():
    assert "error" in task3.get_order_status("9999")




def test_customer_tool():
    result = task3.get_customer("C001")
    assert result["name"] == "Arun"




def test_unknown_customer():
    assert "error" in task3.get_customer("missing")




def test_discount_calculation():
    result = task3.calculate_discount(100, 10)
    assert result["discount"] == 10
    assert result["final_amount"] == 90




def test_invalid_discount():
    assert "error" in task3.calculate_discount(-5, 10)
    assert "error" in task3.calculate_discount(100, 101)




def test_unknown_tool():
    assert "error" in task3.execute_tool("missing_tool", {})




def test_invalid_tool_arguments():
    result = task3.execute_tool("get_customer", {})
    assert "error" in result




def test_tool_round():
    calls = [
        {
            "tool_call_id": "call-1",
            "name": "get_order_status",
            "arguments": {"order_id": "1001"},
        },
        {
            "tool_call_id": "call-2",
            "name": "get_customer",
            "arguments": {"customer_id": "C001"},
        },
    ]


    results = asyncio.run(task3.execute_tool_round(calls))
    assert len(results) == 2
    assert results[0]["tool_call_id"] == "call-1"
    assert results[1]["result"]["name"] == "Arun"




def test_conversation_history_and_summary():
    conversation = task3.Conversation()
    conversation.add("user", "My favourite language is Python")
    conversation.add("assistant", "I will keep that in mind")


    assert len(conversation.full_history()) == 2


    conversation.trim_to_budget(history_budget=1)
    assert isinstance(conversation.summary, str)
    assert "system_budget" in conversation.trim_to_budget(
        history_budget=1
    )




def test_demonstrate_error_cases():
    results = task3.demonstrate_error_cases()
    assert len(results) == 4
    assert all("error" in item for item in results)