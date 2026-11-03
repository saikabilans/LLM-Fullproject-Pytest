from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic

from app.config import settings


def get_openai_client():
    if settings.OPENROUTER_API_KEY:
        return OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    if settings.OPENAI_API_KEY:
        return OpenAI(api_key=settings.OPENAI_API_KEY)

    raise RuntimeError(
        "Add OPENROUTER_API_KEY or OPENAI_API_KEY to .env"
    )


def get_async_openai_client():
    if settings.OPENROUTER_API_KEY:
        return AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    if settings.OPENAI_API_KEY:
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    raise RuntimeError(
        "Add OPENROUTER_API_KEY or OPENAI_API_KEY to .env"
    )


def ask_openai(prompt: str) -> dict:
    client = get_openai_client()

    model = (
        settings.OPENROUTER_MODEL
        if settings.OPENROUTER_API_KEY
        else settings.OPENAI_MODEL
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    usage = response.usage

    return {
        "answer": response.choices[0].message.content or "",
        "model": response.model,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0
    }


def ask_claude(prompt: str, system: str) -> dict:
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "Add ANTHROPIC_API_KEY to .env to use Claude directly"
        )

    client = Anthropic(
        api_key=settings.ANTHROPIC_API_KEY
    )

    response = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", "") == "text"
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return {
        "answer": answer,
        "model": response.model,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    }