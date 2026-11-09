"""Authenticated QQ memory context; no network, model, or automatic writes."""
import base64
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import time

FRAME = re.compile(r"\n<ATRI_QQ_CONTEXT_V1>([A-Za-z0-9_=-]+)\.([0-9a-f]{64})</ATRI_QQ_CONTEXT_V1>")


def read_secret(path):
    secret = Path(path).read_bytes()
    if len(secret) < 32:
        raise ValueError("QQ bridge secret must contain at least 32 bytes")
    return secret


def identity(event):
    platform, sender = str(event.get_platform_id()), str(event.get_sender_id())
    if not platform or not sender or len(platform) > 100 or not sender.isdecimal():
        raise ValueError("verified QQ sender required")
    return "qq:" + platform + ":" + sender


def attach_context(system, owner, records, private, secret, *, now=None, turn=None):
    data = {"version": 1, "owner": owner, "records": records if private else [],
            "private": bool(private), "time": int(time.time() if now is None else now)}
    if turn is not None and private:
        data["turn"] = turn
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(raw).decode()
    signature = hmac.new(secret, encoded.encode(), hashlib.sha256).hexdigest()
    return FRAME.sub("", system or "") + f"\n<ATRI_QQ_CONTEXT_V1>{encoded}.{signature}</ATRI_QQ_CONTEXT_V1>"


def detach_context(messages, secret, *, now=None):
    result = [dict(m) for m in messages]
    found = []
    for message in result:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        matches = list(FRAME.finditer(message["content"]))
        if message["content"].count("<ATRI_QQ_CONTEXT_V1>") != len(matches):
            raise ValueError("malformed QQ context")
        for match in matches:
            found.append(match.groups())
        message["content"] = FRAME.sub("", message["content"])
    if not found:
        return result, None
    if len(found) != 1:
        raise ValueError("ambiguous QQ context")
    encoded, signature = found[0]
    expected = hmac.new(secret, encoded.encode(), hashlib.sha256).hexdigest()
    if len(encoded) > 131072 or not hmac.compare_digest(signature, expected):
        raise ValueError("invalid QQ context")
    try:
        data = json.loads(base64.urlsafe_b64decode(encoded))
        age = (time.time() if now is None else now) - data["time"]
        if data["version"] != 1 or not -10 <= age <= 180:
            raise ValueError("expired context")
        if not isinstance(data["private"], bool) or not isinstance(data["owner"], str):
            raise ValueError("invalid identity")
        if not data["owner"].startswith("qq:") or len(data["owner"]) > 220:
            raise ValueError("invalid identity")
        rows = data["records"]
        if not isinstance(rows, list) or len(rows) > 24:
            raise ValueError("invalid records")
        if not all(isinstance(row, dict) and set(row) == {"key", "value"}
                   and isinstance(row["key"], str) and 0 < len(row["key"]) <= 64
                   and isinstance(row["value"], str) and 0 < len(row["value"]) <= 400 for row in rows):
            raise ValueError("invalid records")
        if not data["private"] and rows:
            raise ValueError("group context cannot contain private records")
        if "turn" in data:
            turn = data["turn"]
            if (not data["private"] or not isinstance(turn, dict)
                    or set(turn) != {"conversation_id", "sent_at", "previous_at"}
                    or not isinstance(turn["conversation_id"], str)
                    or not 0 < len(turn["conversation_id"]) <= 128
                    or any(type(turn[k]) not in (int, float) or not math.isfinite(turn[k])
                           or turn[k] < 0 for k in ("sent_at", "previous_at"))
                    or turn["sent_at"] <= 0
                    or turn["sent_at"] > data["time"] + 10):
                raise ValueError("invalid conversation clock")
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("invalid or expired QQ context") from exc
    return result, data
