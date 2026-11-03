
## Project Overview

This project implements an LLM application using Python, FastAPI, and OpenRouter. It covers LLM API calls, streaming responses, retries, asynchronous execution, tool calling, conversation memory, API security, and automated testing.

## Technologies Used

- Python
- FastAPI
- OpenAI Python SDK
- OpenRouter API
- Pydantic
- Tenacity
- Tiktoken
- Pytest
- HTTPX

## Project Structure

```text
LLM_Fullproject/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── llm.py
│   ├── main.py
│   ├── task1.py
│   ├── task2.py
│   ├── task3.py
│   └── task4.py
│
├── tests/
│   ├── test_api.py
│   └── test_task4.py
│
├── logs/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Features

### Task 1: SDK Calls, Streaming and Cost
- Load API configuration from environment variables.
- Make LLM API requests.
- Support streaming responses.
- Count tokens and track usage.

### Task 2: Reliability and Async
- Retry temporary provider errors.
- Use exponential backoff and jitter.
- Compare sequential and concurrent requests.
- Limit concurrent requests using an asyncio semaphore.

### Task 3: Tools and Conversation Memory
- Implement tool calling.
- Handle tool execution results and errors.
- Maintain conversation history.
- Manage conversation context.

### Task 4: Secured FastAPI Backend
- Create and manage chat sessions.
- Validate API keys.
- Expose API endpoints.
- Handle errors and validate requests.
- Test API behavior using pytest.

## Setup Instructions

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
```

### 2. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

Add your own API key and configuration values. For example:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
API_KEY=your_local_api_secret
```

Use the exact variable names expected by `app/config.py`. Never commit your actual `.env` file to GitHub.

## Run the Application

Start the FastAPI server:

```powershell
uvicorn app.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Health endpoint:

http://127.0.0.1:8000/health

## Run Tests

Run all tests:

```powershell
python -m pytest -v
```

Run tests with coverage:

```powershell
python -m pytest --cov=app --cov-report=term-missing
```

The current test coverage is approximately 71%. Coverage may vary depending on the tests and code executed.

## Security Notes

- Keep API keys private.
- Do not upload `.env` to GitHub.
- Use `.env.example` to document required environment variables without real secrets.
- Keep authentication enabled for protected API endpoints.

## Author

LLM Application Development Assessment 