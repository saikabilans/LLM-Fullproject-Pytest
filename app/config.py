import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    API_KEY = os.getenv("API_KEY", "my-local-secret-123")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv(
        "ANTHROPIC_MODEL",
        "claude-3-5-haiku-latest"
    )

    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:5500"
        ).split(",")
        if origin.strip()
    ]


settings = Settings()