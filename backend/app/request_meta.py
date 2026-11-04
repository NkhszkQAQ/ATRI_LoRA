from __future__ import annotations

from typing import Any

from fastapi import Request

from app.security import safe_decode


def client_ip(request: Request) -> str:
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    real = (request.headers.get("x-real-ip") or "").strip()
    if real:
        return real
    return request.client.host if request.client else ""


def client_ua(request: Request) -> str:
    return (request.headers.get("user-agent") or "")[:1000]


def auth_token_info(request: Request) -> dict[str, Any]:
    """Decode JWT claims for audit (never store raw bearer token)."""
    auth = request.headers.get("authorization") or ""
    info: dict[str, Any] = {
        "scheme": "bearer" if auth.lower().startswith("bearer ") else ("none" if not auth else "other"),
        "header_present": bool(auth),
    }
    if auth.lower().startswith("bearer "):
        raw = auth[7:].strip()
        info["token_prefix"] = raw[:12] + "…" if len(raw) > 12 else raw
        info["token_length"] = len(raw)
        payload = safe_decode(raw)
        if payload:
            claims = {}
            for k, v in payload.items():
                if k in ("sub", "exp", "iat", "nbf", "jti", "iss", "aud", "type", "role"):
                    claims[k] = v
            info["jwt_claims"] = claims
        else:
            info["jwt_claims"] = None
            info["jwt_decode"] = "invalid_or_expired"
    return info


def build_token_info(
    request: Request,
    *,
    model: str = "",
    enable_thinking: bool | None = None,
    usage: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "auth": auth_token_info(request),
        "model": model,
        "usage": usage
        or {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "note": "未计量（拦截未走模型或上游未返回 usage）",
        },
    }
    if enable_thinking is not None:
        data["enable_thinking"] = enable_thinking
    if extra:
        data.update(extra)
    return data
