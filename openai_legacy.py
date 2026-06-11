from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import openai


def get_openai_api_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def concise_openai_error(exc: Exception, *, max_chars: int = 240) -> str:
    message = " ".join(str(exc).split()) or "OpenAI request failed"
    if len(message) > max_chars:
        message = message[: max_chars - 3] + "..."
    return f"{type(exc).__name__}: {message}"


def chat_completion_text(
    *,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, str]:
    """Return text and an error string using the legacy OpenAI 0.28 chat API."""
    key = (api_key or get_openai_api_key()).strip()
    if not key:
        return "", "missing_api_key"

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    try:
        openai.api_key = key
        response = openai.ChatCompletion.create(**kwargs)
        text = str(response["choices"][0]["message"]["content"] or "").strip()
        if not text:
            return "", "empty_response"
        return text, ""
    except Exception as exc:
        return "", f"openai_call_failed: {concise_openai_error(exc)}"
