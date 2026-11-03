import hmac
import uuid

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import ask_openai
from app.task1 import router as task1_router


app = FastAPI(title="Week 5 LLM Assessment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS[:2],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "x-api-key"]
)

app.include_router(task1_router)

SESSIONS: dict[str, list[dict[str, str]]] = {}


class ChatInput(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def require_api_key(
    x_api_key: str | None = Header(default=None)
):
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail={"error": "API key missing"}
        )

    if not hmac.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=403,
            detail={"error": "Invalid API key"}
        )

    return True


def get_settings():
    return settings


def get_llm():
    return ask_openai


def get_store():
    return SESSIONS


@app.exception_handler(HTTPException)
async def handle_http_error(
    request: Request,
    error: HTTPException
):
    if isinstance(error.detail, dict):
        body = error.detail
    else:
        body = {"error": str(error.detail)}

    return JSONResponse(
        status_code=error.status_code,
        content=body,
        headers=error.headers
    )


@app.get("/health")
def health():
    # Public endpoint; no API key is required.
    return {"status": "ok"}


@app.post(
    "/sessions",
    dependencies=[Depends(require_api_key)]
)
async def create_session(
    store=Depends(get_store),
    config=Depends(get_settings)
):
    session_id = str(uuid.uuid4())
    store[session_id] = []

    return {"session_id": session_id}


@app.post(
    "/sessions/{session_id}/chat",
    dependencies=[Depends(require_api_key)]
)
async def chat_in_session(
    session_id: str,
    request: ChatInput,
    store=Depends(get_store),
    llm=Depends(get_llm)
):
    if session_id not in store:
        raise HTTPException(
            status_code=404,
            detail={"error": "Session not found"}
        )

    store[session_id].append({
        "role": "user",
        "content": request.message
    })

    try:
        result = llm(request.message)

        answer = (
            result.get("answer", "")
            if isinstance(result, dict)
            else str(result)
        )

    except Exception as error:
        store[session_id].pop()

        raise HTTPException(
            status_code=502,
            detail={"error": "LLM service failed"}
        ) from error

    store[session_id].append({
        "role": "assistant",
        "content": answer
    })

    return {
        "session_id": session_id,
        "answer": answer
    }


@app.get(
    "/sessions/{session_id}/messages",
    dependencies=[Depends(require_api_key)]
)
async def get_session_messages(
    session_id: str,
    store=Depends(get_store)
):
    if session_id not in store:
        raise HTTPException(
            status_code=404,
            detail={"error": "Session not found"}
        )

    return {
        "session_id": session_id,
        "messages": store[session_id]
    }