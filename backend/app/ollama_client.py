from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from app.config import get_settings


async def list_models() -> list[str]:
    settings = get_settings()
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=30.0) as client:
        r = await client.get("/api/tags")
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


async def chat_stream(
    *,
    model: str,
    messages: list[dict[str, str]],
    enable_thinking: bool = True,
) -> AsyncIterator[dict[str, str]]:
    """Yield dicts: {"content": "...", "reasoning": "..."}."""
    settings = get_settings()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": enable_thinking,
        "enable_thinking": enable_thinking,
        "options": {"temperature": 0.85, "top_p": 0.9, "num_predict": 1024},
    }
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=None) as client:
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                import json

                obj = json.loads(line)
                msg = obj.get("message") or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning") or ""
                # some servers put thinking under different keys
                if not reasoning:
                    reasoning = msg.get("thinking") or ""
                if content or reasoning:
                    yield {"content": content, "reasoning": reasoning}
                if obj.get("done"):
                    break


async def chat_once(*, model: str, messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
    parts: list[str] = []
    async for c in chat_stream(model=model, messages=messages, enable_thinking=enable_thinking):
        if c.get("content"):
            parts.append(c["content"])
    return "".join(parts)
