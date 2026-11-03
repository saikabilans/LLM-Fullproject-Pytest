import json
import time
from datetime import date
from pathlib import Path

import tiktoken
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import ask_openai, get_openai_client

router = APIRouter()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
USAGE_FILE = LOG_DIR / "usage.jsonl"

MAX_INPUT_TOKENS = 3000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def count_tokens(messages: list[dict]) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 0

    for message in messages:
        total += 4
        for value in message.values():
            if isinstance(value, str):
                total += len(encoding.encode(value))

    return total + 2


def log_usage(record: dict):
    with USAGE_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def get_today_usage():
    total_tokens = 0
    total_cost = 0.0

    if USAGE_FILE.exists():
        for line in USAGE_FILE.read_text(
            encoding="utf-8"
        ).splitlines():
            try:
                row = json.loads(line)

                if row.get("date") == date.today().isoformat():
                    total_tokens += int(row.get("total_tokens", 0))
                    total_cost += float(row.get("cost_usd", 0))

            except (ValueError, TypeError):
                continue

    return {
        "date": date.today().isoformat(),
        "total_tokens": total_tokens,
        "cost_usd": round(total_cost, 6)
    }


@router.post("/chat")
def chat(request: ChatRequest):
    messages = [
        {"role": "user", "content": request.message}
    ]

    if count_tokens(messages) > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=413,
            detail="Input exceeds token limit"
        )

    started = time.perf_counter()

    try:
        result = ask_openai(request.message)
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="LLM call failed"
        ) from error

    seconds = round(time.perf_counter() - started, 3)

    # Example rates only. Check the selected model's current pricing.
    cost = (
        result["prompt_tokens"] * 0.00000015
        + result["completion_tokens"] * 0.00000060
    )

    log_usage({
        "date": date.today().isoformat(),
        "model": result["model"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "cost_usd": cost,
        "seconds": seconds
    })

    return {
        **result,
        "seconds": seconds,
        "estimated_cost_usd": round(cost, 6)
    }


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    messages = [
        {"role": "user", "content": request.message}
    ]

    if count_tokens(messages) > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=413,
            detail="Input exceeds token limit"
        )

    def events():
        try:
            client = get_openai_client()

            model = (
                settings.OPENROUTER_MODEL
                if settings.OPENROUTER_API_KEY
                else settings.OPENAI_MODEL
            )

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )

            for chunk in stream:
                text = chunk.choices[0].delta.content

                if text:
                    yield f"data: {json.dumps({'text': text})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as error:
            yield (
                "data: "
                + json.dumps({"error": type(error).__name__})
                + "\n\n"
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream"
    )


@router.get("/usage")
def usage():
    return get_today_usage()