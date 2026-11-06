#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATRI v1.3 service: Qwen3-8B + existing v1.1-R LoRA, optional grounded answers.

No retraining or adapter version change. See docs/V1_3.md for the runtime profile.
"""

import json
import os
import re
import threading
import time
import uuid
import asyncio

BASE = os.environ.get("ATRI_BASE", "/root/autodl-tmp/models/models/Qwen--Qwen3-8B")
ADAPTER = os.environ.get("ATRI_ADAPTER", "/root/autodl-tmp/atri-v1.1-8b/adapter")
SYSTEM_FILE = os.environ.get("ATRI_SYSTEM", "/root/autodl-tmp/atri-system.txt")
SEED = int(os.environ.get("ATRI_SEED", "20261104"))
PORT = int(os.environ.get("ATRI_PORT", "8003"))
MODEL_NAME = os.environ.get("ATRI_MODEL_NAME", "atri-v1.3-8B")
RUNTIME_VERSION = "1.3.0"

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
        with open(SYSTEM_FILE, encoding="utf-8") as source:
            text = source.read().strip()
        return text.replace("{{user}}", "主人").replace("{{honorific}}", "").replace(
            "{{memory}}", "（长期记忆由服务按当前用户单独注入；未提供的信息不能编造）")
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
        if (os.getenv("ATRI_GROUNDING_ENABLED", "1").lower() in {"1", "true", "yes"}
                and not getattr(rag, "grounding_ready", True)):
            raise ValueError("v1.3 grounded answers require an index rebuilt with --facts; use a new index directory")
    memory = None
    if os.getenv("ATRI_MEMORY_ENABLED", "0").lower() in {"1", "true", "yes"}:
        from atri_memory import from_env
        memory = from_env()
        memory.install(app)

    @app.get("/health")
    def health():
        return {"ok": "model" in state, "model": MODEL_NAME, "runtime_version": RUNTIME_VERSION,
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

        original_msgs = [dict(m) for m in msgs]
        tasks = []
        grounded = os.getenv("ATRI_GROUNDING_ENABLED", "1").lower() in {"1", "true", "yes"}
        if grounded and (rag is not None or memory is not None):
            from atri_grounding import make_plan
            records = await asyncio.to_thread(memory.store.list, owner) if memory else None
            tasks = await asyncio.to_thread(make_plan, original_msgs, rag, records)
        if not tasks:
            if rag is not None:
                msgs = await asyncio.to_thread(rag.augment, msgs)
            if memory is not None:
                msgs = await asyncio.to_thread(memory.store.augment, owner, msgs)

        def clean_answer(text):
            if grounded and (rag is not None or memory is not None):
                from atri_grounding import normalize_output
                context = "\n".join(m["content"] for m in msgs if m["role"] == "user")
                return normalize_output(text, context)
            return text

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

        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        want_stream = bool(body.get("stream"))

        if tasks:
            from atri_grounding import render_plan
            attempts = []
            per_task_limit = max(1, max_tokens // len(tasks))

            def grounded_generate(plan_messages):
                plan_text = tok.apply_chat_template(plan_messages, tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
                plan_inputs = tok(plan_text, return_tensors="pt").to(model.device)
                size = plan_inputs["input_ids"].shape[1]
                options = {**gen_kwargs, "max_new_tokens": per_task_limit}
                with lock:
                    if isinstance(seed, int):
                        torch.manual_seed(seed)
                    with torch.no_grad():
                        output = model.generate(**plan_inputs, **options)
                attempts.append({"prompt": size, "completion": output.shape[1] - size})
                return tok.decode(output[0][size:], skip_special_tokens=True).strip()

            system = "\n".join(m["content"] for m in original_msgs if m["role"] == "system")
            prepare_chat = (lambda messages: memory.store.augment(owner, messages)) if memory else None
            answer, trace = await asyncio.to_thread(render_plan, tasks, system, grounded_generate,
                                                   original_msgs[:-1], prepare_chat)
            answer_ids = tok.encode(answer, add_special_tokens=False)
            finish = "length" if len(answer_ids) > max_tokens else "stop"
            if finish == "length":
                answer_ids = answer_ids[:max_tokens]
                answer = tok.decode(answer_ids, skip_special_tokens=True)
            usage = {"prompt_tokens": sum(a["prompt"] for a in attempts) or prompt_len,
                     "completion_tokens": len(answer_ids)}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            details = {"runtime_version": RUNTIME_VERSION, "tasks": trace,
                       "generation_attempts": len(attempts),
                       "internal_completion_tokens": sum(a["completion"] for a in attempts)}
            if not want_stream:
                return {"id": rid, "object": "chat.completion", "created": created,
                        "model": MODEL_NAME, "choices": [{"index": 0, "message": {
                            "role": "assistant", "content": answer}, "finish_reason": finish}],
                        "usage": usage, "atri": details}

            async def checked_stream():
                # Fact/memory tasks are buffered until validated. Never send a
                # contradicted draft to the client and try to retract it later.
                for offset in range(0, len(answer), 32):
                    delta = {"content": answer[offset:offset + 32]}
                    if offset == 0:
                        delta["role"] = "assistant"
                    chunk = {"id": rid, "object": "chat.completion.chunk", "created": created,
                             "model": MODEL_NAME, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                chunk = {"id": rid, "object": "chat.completion.chunk", "created": created,
                         "model": MODEL_NAME, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                         "atri": details}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(checked_stream(), media_type="text/event-stream")

        if not want_stream:
            with lock:
                if isinstance(seed, int):
                    torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
            ans = clean_answer(tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip())
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
                if isinstance(seed, int):
                    torch.manual_seed(seed)
                with torch.no_grad():
                    model.generate(**inputs, streamer=streamer, **gen_kwargs)

        def sse(obj) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        async def gen():
            th = threading.Thread(target=run, daemon=True)
            th.start()
            first = True
            pending = ""
            def content_chunk(piece, initial):
                return {"id": rid, "object": "chat.completion.chunk", "created": created,
                        "model": MODEL_NAME, "choices": [{"index": 0, "delta": {
                            **({"role": "assistant"} if initial else {}), "content": piece}, "finish_reason": None}]}
            for piece in streamer:
                if not piece:
                    continue
                if grounded and (rag is not None or memory is not None):
                    pending += piece
                    boundaries = list(re.finditer(r"[。！？\n]", pending))
                    if not boundaries:
                        continue
                    end = boundaries[-1].end()
                    piece, pending = clean_answer(pending[:end]), pending[end:]
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
            if pending:
                yield sse(content_chunk(clean_answer(pending), first))
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
    application = build_app()
    load()
    import uvicorn

    uvicorn.run(application, host="127.0.0.1", port=PORT, log_level="info")
