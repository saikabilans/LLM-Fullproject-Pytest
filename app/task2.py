import asyncio
import logging
import random
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception
)

logger = logging.getLogger("task2")


class ProviderError(Exception):
    def __init__(self, status_code: int, message="Provider error"):
        super().__init__(message)
        self.status_code = status_code


def should_retry(error):
    return isinstance(error, ProviderError) and (
        error.status_code == 429
        or 500 <= error.status_code <= 599
    )


@retry(
    retry=retry_if_exception(should_retry),
    wait=wait_exponential(multiplier=1, min=1, max=8)
         + wait_random(0, 1),
    stop=stop_after_attempt(5),
    before_sleep=lambda state: logger.warning(
        "Attempt %s failed; waiting before retry",
        state.attempt_number
    ),
    reraise=True
)
def retrying_call(function):
    return function()


async def ask_openai_async(prompt: str) -> dict:
    from app.llm import get_async_openai_client
    from app.config import settings

    client = get_async_openai_client()

    model = (
        settings.OPENROUTER_MODEL
        if settings.OPENROUTER_API_KEY
        else settings.OPENAI_MODEL
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    usage = response.usage

    return {
        "answer": response.choices[0].message.content or "",
        "model": response.model,
        "total_tokens": usage.total_tokens if usage else 0
    }


async def compare_sequential_and_parallel(question_function):
    questions = [
        f"Question {number}" for number in range(1, 11)
    ]

    # Run sequentially.
    started = time.perf_counter()
    sequential = []

    for question in questions:
        try:
            sequential.append(await question_function(question))
        except Exception as error:
            sequential.append({"error": str(error)})

    sequential_time = time.perf_counter() - started

    # Run concurrently, with at most three active requests.
    semaphore = asyncio.Semaphore(3)

    async def safe_call(question):
        async with semaphore:
            try:
                return await question_function(question)
            except Exception as error:
                return {"error": str(error)}

    started = time.perf_counter()

    parallel = await asyncio.gather(
        *(safe_call(question) for question in questions)
    )

    parallel_time = time.perf_counter() - started

    return {
        "sequential_seconds": round(sequential_time, 2),
        "parallel_seconds": round(parallel_time, 2),
        "sequential_results": sequential,
        "parallel_results": parallel
    }


def backoff_explanation():
    return [
        "Exponential backoff increases the wait after each failure.",
        "Jitter adds a small random delay to reduce simultaneous retries.",
        "HTTP 429 and 5xx errors may be temporary.",
        "HTTP 401 is an authentication error and should not be retried."
    ]


def demo_retry():
    attempts = {"count": 0}

    def sometimes_fails():
        attempts["count"] += 1

        if attempts["count"] < 3:
            raise ProviderError(503, "Temporary server error")

        return "Success after retries"

    result = retrying_call(sometimes_fails)

    return {
        "result": result,
        "attempts": attempts["count"]
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Retry demonstration:")
    print(demo_retry())

    print("\nBackoff explanation:")
    for item in backoff_explanation():
        print("-", item)

    print("\nTo compare 10 real async requests, call:")
    print("asyncio.run(compare_sequential_and_parallel(ask_openai_async))")