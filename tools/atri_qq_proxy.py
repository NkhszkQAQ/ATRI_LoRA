"""CPU RAG/memory bridge in front of an existing local ATRI model gateway.

The model processes, adapter switching, and AstrBot conversation DB stay separate.
Only the AstrBot event plugin can supply signed private memory context.
"""
import argparse
import asyncio
import json
import re
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from atri_bridge import detach_context, read_secret
from atri_context import ContextWindow
from atri_grounding import VOICE, make_plan, normalize_output, render_plan
from atri_memory import augment_records

VERSION = "1.3.0"
RULES = ("\n[核实后的事实与记忆边界]\n亚托莉能进食、品尝味道，但不能通过进食补充营养。"
         "如旧人设写成不能吃东西，以此处核实后的边界纠正。未核实的充电、接口等规格不猜测。"
         "用户记忆属于当前提问用户，不是ATRI自己的喜好或经历。多个问题逐项回答，"
         "原作事实不能用自夸替代，不把夏生的经历写成当前用户的经历。")
RULES += VOICE


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(x.get("text", "") for x in value if isinstance(x, dict) and x.get("type") == "text")
    return ""


def add_system(messages, text):
    result = [dict(m) for m in messages]
    system = next((m for m in result if m.get("role") == "system" and isinstance(m.get("content"), str)), None)
    if system is None:
        result.insert(0, {"role": "system", "content": text})
    else:
        system["content"] += text
    return result


def build_app(config, *, rag=None, client=None, tokenizer=None):
    secret = read_secret(config["shared_secret_file"])
    context_options = config.get("context_window", {})
    context_path = config.get("context_db")
    if context_path is None and config.get("memory_db"):
        context_path = str(Path(config["memory_db"]).with_name("qq-context.sqlite3"))
    if context_path and config.get("memory_db") and Path(context_path).resolve() == Path(config["memory_db"]).resolve():
        raise ValueError("context_db must be separate from memory_db")
    window = ContextWindow(context_path, **context_options)
    if rag is None:
        from atri_rag import Config, Retriever
        rag = Retriever(Path(config["rag_storage"]), Config())
    if not getattr(rag, "grounding_ready", False):
        raise ValueError("QQ RAG requires the v1.3 fact index")
    if not getattr(rag, "voice_ready", False):
        raise ValueError("QQ RAG requires a fact index rebuilt with spoken_answer")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True, trust_remote_code=False)
    client = client or httpx.Client(timeout=180, trust_env=False)
    gateway = config["gateway"].rstrip("/")
    backends = config["backends"]
    rag_model = config.get("rag_model", "v1.1-8B")
    app = FastAPI()
    counts = {"signed_requests": 0, "anonymous_requests": 0, "planned_requests": 0}
    app.state.counts = counts

    def current_model():
        try:
            response = client.get(gateway + "/models", timeout=5)
            response.raise_for_status()
            name = response.json()["current"]
            if name not in backends:
                raise ValueError("unknown gateway model")
            return name
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise HTTPException(503, "model selection gateway unavailable") from exc

    def upstream(name, payload):
        response = client.post(backends[name].rstrip("/") + "/v1/chat/completions", json=payload)
        if response.is_error:
            raise HTTPException(response.status_code, "model backend rejected request")
        return response

    @app.get("/health")
    def health():
        return {"ok": True, "runtime_version": VERSION, "rag_model": rag_model,
                "working_context": {"max_history_turns": window.max_turns, "idle_timeout_seconds": window.idle_seconds},
                "answer_voice": "first_person",
                "memory": "signed private QQ context", "counts": dict(counts)}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "atri"} for name in backends]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        raw = body.get("messages")
        if not isinstance(raw, list) or not raw or any(not isinstance(m, dict) for m in raw):
            raise HTTPException(400, "messages required")
        try:
            messages, context = detach_context(raw, secret)
        except ValueError as exc:
            raise HTTPException(401, "invalid QQ memory context") from exc
        counts["signed_requests" if context else "anonymous_requests"] += 1
        messages, _ = await asyncio.to_thread(window.select, messages, context)
        records = context["records"] if context and context["private"] else None
        notice = None
        if context and not context["private"]:
            notice = "这件事到私聊再说吧，主人。群里我不会翻你的私人记忆。"
        elif context is None and re.search(r"保存|长期记忆", text_content(messages[-1].get("content"))):
            notice = "先在QQ私聊里问我吧，主人。我得认得是谁，才能拿出对应的记忆。"
        name = await asyncio.to_thread(current_model)
        active_rag = rag if name == rag_model else None
        messages = [{**m, "content": text_content(m["content"])}
                    if isinstance(m.get("content"), list) and all(isinstance(p, dict) and p.get("type") == "text" for p in m["content"])
                    else m for m in messages]
        if active_rag:
            messages = add_system(messages, RULES)
        flat = [{**m, "content": text_content(m.get("content"))} for m in messages]
        # Preserve media / tool conversations through the old model path.
        plain = all(isinstance(m.get("content"), str) and m.get("role") in {"system", "user", "assistant"} for m in messages)
        tasks = await asyncio.to_thread(make_plan, flat, active_rag, records,
                                        external_memory=True, memory_notice=notice) if plain else []
        payload = {**body, "model": name, "messages": messages}
        if not tasks:
            if active_rag and plain:
                messages = await asyncio.to_thread(active_rag.augment, messages)
            if records is not None:
                messages = augment_records(messages, records)
            payload["messages"] = messages
            if not body.get("stream"):
                response = await asyncio.to_thread(upstream, name, payload)
                value = response.json()
                if active_rag:
                    known = "\n".join(text_content(m.get("content")) for m in messages if m.get("role") == "user")
                    value["choices"][0]["message"]["content"] = normalize_output(value["choices"][0]["message"]["content"], known)
                return value
            # Establish upstream status before returning an HTTP 200 SSE stream.
            upstream_request = client.build_request("POST", backends[name].rstrip("/") + "/v1/chat/completions", json=payload)
            response = await asyncio.to_thread(client.send, upstream_request, stream=True)
            if response.is_error:
                status = response.status_code
                response.close()
                raise HTTPException(status, "model backend rejected stream")
            def passthrough():
                try:
                    yield from response.iter_bytes()
                finally:
                    response.close()
            return StreamingResponse(passthrough(), media_type="text/event-stream")

        counts["planned_requests"] += 1
        try:
            limit = max(1, min(int(body.get("max_tokens") or body.get("max_new_tokens") or 512), 2048))
        except (TypeError, ValueError):
            limit = 512
        attempts = []
        per_task = max(1, limit // len(tasks))
        def generate(msgs):
            response = upstream(name, {**payload, "messages": msgs, "stream": False, "max_tokens": per_task})
            value = response.json()
            attempts.append(value.get("usage", {}))
            return value["choices"][0]["message"]["content"]
        system = "\n".join(m["content"] for m in flat if m.get("role") == "system")
        prepare = (lambda msgs: augment_records(msgs, records)) if records is not None else None
        answer, trace = await asyncio.to_thread(render_plan, tasks, system, generate, flat[:-1], prepare)
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        finish = "length" if len(tokens) > limit else "stop"
        if finish == "length":
            tokens = tokens[:limit]
            answer = tokenizer.decode(tokens, skip_special_tokens=True)
        details = {"runtime_version": VERSION, "tasks": trace, "generation_attempts": len(attempts),
                   "internal_completion_tokens": sum(a.get("completion_tokens", 0) for a in attempts)}
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        common = {"id": rid, "created": created, "model": name}
        if not body.get("stream"):
            prompt_tokens = sum(a.get("prompt_tokens", 0) for a in attempts)
            return {**common, "object": "chat.completion", "choices": [{"index": 0, "message": {
                "role": "assistant", "content": answer}, "finish_reason": finish}], "atri": details,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(tokens),
                          "total_tokens": prompt_tokens + len(tokens)}}
        def checked_stream():
            for i in range(0, len(answer), 32):
                delta = {"content": answer[i:i + 32], **({"role": "assistant"} if i == 0 else {})}
                yield "data: " + json.dumps({**common, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": delta, "finish_reason": None}]}, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({**common, "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {}, "finish_reason": finish}], "atri": details}, ensure_ascii=False) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(checked_stream(), media_type="text/event-stream")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    import os
    for key, value in settings.get("environment", {}).items():
        if key not in {"HF_HOME", "HF_HUB_OFFLINE", "TIKTOKEN_CACHE_DIR", "DISABLE_SAFETENSORS_CONVERSION", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM"}:
            raise ValueError("unsupported runtime environment setting")
        os.environ[key] = str(value)
    import uvicorn
    uvicorn.run(build_app(settings), host="127.0.0.1", port=int(settings.get("port", 9013)))
