#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATRI v1.0-R 推理服务（OpenAI 兼容）—— 新文件夹专用，不动原 serve.py

跑法（3080 上）：
    /root/autodl-tmp/astrbot/venv/bin/python /root/autodl-tmp/atri-v1r/serve_v1r.py

接口：
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions      # 支持 stream，content 可以是字符串或数组

参数（按主人要求）：
    - 模型：ATRI v1.0-R（GitHub release v1.0-R）
    - 不用 greedy：temperature 0.7 / top_p 0.9 采样
    - 固定种子 20261104
"""

import json
import os
import threading
import time
import uuid
import asyncio

BASE = os.environ.get("ATRI_BASE", "/root/autodl-tmp/models/models/Qwen--Qwen3-8B")
ADAPTER = os.environ.get("ATRI_ADAPTER", "/root/autodl-tmp/atri-v1.1-8b/adapter")
SYSTEM_FILE = os.environ.get("ATRI_SYSTEM", "/root/autodl-tmp/atri-system.txt")
SEED = int(os.environ.get("ATRI_SEED", "20261104"))
PORT = int(os.environ.get("ATRI_PORT", "8003"))
MODEL_NAME = os.environ.get("ATRI_MODEL_NAME", "atri-v1.1-8B")

state = {}
lock = threading.Lock()


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    t0 = time.time()
    print(f"[init] base={BASE}", flush=True)
    print(f"[init] adapter={ADAPTER}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    state["tok"] = tok
    state["model"] = model
    print(f"[init] ready in {time.time() - t0:.1f}s", flush=True)


def default_system() -> str:
    if os.path.exists(SYSTEM_FILE):
        return open(SYSTEM_FILE, encoding="utf-8").read().strip()
    return ""


def flat_content(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        buf = []
        for part in c:
            if isinstance(part, str):
                buf.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    buf.append(part.get("text") or "")
                elif isinstance(part.get("content"), str):
                    buf.append(part["content"])
        return "".join(buf)
    return str(c)


def build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    # Preserve standalone script use when the optional RAG feature is disabled.
    rag = None
    if os.getenv("ATRI_RAG_ENABLED", "0").lower() in {"1", "true", "yes"}:
        from atri_rag import from_env
        rag = from_env()
    memory = None
    if os.getenv("ATRI_MEMORY_ENABLED", "0").lower() in {"1", "true", "yes"}:
        from atri_memory import from_env
        memory = from_env()
        memory.install(app)

    @app.get("/health")
    def health():
        return {"ok": "model" in state, "model": MODEL_NAME,
                "seed": "请求可传 seed，默认随机"}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "atri",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        owner = memory.owner(req.headers.get("authorization")) if memory else None
        body = await req.json()
        raw_msgs = body.get("messages") or []
        msgs = []
        for m in raw_msgs:
            if isinstance(m, dict):
                msgs.append(
                    {
                        "role": m.get("role") or "user",
                        "content": flat_content(m.get("content")),
                    }
                )
        if not msgs:
            return JSONResponse({"error": "messages required"}, status_code=400)

        if memory is not None:
            reply = None
            if msgs[-1]["role"] == "user":
                try:
                    reply = await asyncio.to_thread(memory.store.command, owner, msgs[-1]["content"])
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)
            if reply is not None:
                response = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            "created": int(time.time()), "model": MODEL_NAME}
                if body.get("stream"):
                    async def command_stream():
                        for delta, finish in [({"role": "assistant", "content": reply}, None), ({}, "stop")]:
                            chunk = {**response, "object": "chat.completion.chunk", "choices": [
                                {"index": 0, "delta": delta, "finish_reason": finish}]}
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(command_stream(), media_type="text/event-stream")
                return {**response, "object": "chat.completion", "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}

        if not any(m["role"] == "system" for m in msgs):
            sys_text = default_system()
            if sys_text:
                msgs.insert(0, {"role": "system", "content": sys_text})

        if rag is not None:
            msgs = await asyncio.to_thread(rag.augment, msgs)
        if memory is not None:
            msgs = await asyncio.to_thread(memory.store.augment, owner, msgs)

        import torch
        from transformers import TextIteratorStreamer

        tok, model = state["tok"], state["model"]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tok(text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 512
        try:
            max_tokens = max(1, min(int(max_tokens), 2048))
        except (TypeError, ValueError):
            max_tokens = 512

        t = body.get("temperature")
        temperature = float(t) if isinstance(t, (int, float)) else 0.7
        tp = body.get("top_p")
        top_p = float(tp) if isinstance(tp, (int, float)) else 0.9
        # 主人要求：调试不要开 greedy，固定种子
        do_sample = True
        if temperature <= 0:
            temperature = 0.7

        seed = body.get("seed")

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            use_cache=True,
        )

        if isinstance(seed, int):
            torch.manual_seed(seed)   # 用户传了才固定；不传则每轮随机

        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        want_stream = bool(body.get("stream"))

        if not want_stream:
            with lock:
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
            ans = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
            return {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ans or "……"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": out.shape[1] - prompt_len,
                    "total_tokens": out.shape[1],
                },
            }

        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )

        def run():
            with lock:
                with torch.no_grad():
                    model.generate(**inputs, streamer=streamer, **gen_kwargs)

        def sse(obj) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        async def gen():
            th = threading.Thread(target=run, daemon=True)
            th.start()
            first = True
            for piece in streamer:
                if not piece:
                    continue
                delta = (
                    {"role": "assistant", "content": piece}
                    if first
                    else {"content": piece}
                )
                first = False
                yield sse(
                    {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
            yield sse(
                {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


if __name__ == "__main__":
    load()
    import uvicorn

    uvicorn.run(build_app(), host="127.0.0.1", port=PORT, log_level="info")
