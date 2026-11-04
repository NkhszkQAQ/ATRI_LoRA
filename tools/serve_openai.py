#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATRI LoRA —— OpenAI 兼容推理服务（给 AstrBot 等外部程序调用）

跑在 GPU 服务器上：
    /root/autodl-tmp/atri-env/bin/python /root/autodl-tmp/atri/serve_openai.py \
        --adapter v13 --port 8000

接口（OpenAI 兼容）：
    GET  /v1/models
    POST /v1/chat/completions     # 支持 stream

设计说明：
- system prompt 由调用方（AstrBot 的 persona）传，服务端不再叠加，避免双份人设。
- 单卡串行：一把锁保护 generate，避免并发把显存/上下文搞乱。
- 4bit 量化加载 Qwen3-4B + LoRA adapter，和 atri/chat.py 完全同一套加载方式。
"""

import argparse
import json
import os
import sys
import threading
import time
import uuid

ADAPTERS = {
    "v5": "/root/autodl-tmp/atri-runs/sft-v5-deep",
    "v10": "/root/autodl-tmp/atri-runs/sft-v10-multiturn",
    "v11": "/root/autodl-tmp/atri-runs/sft-v11-mixed",
    "v12": "/root/autodl-tmp/atri-runs/sft-v12-mixed",
    "v13": "/root/autodl-tmp/atri-runs/sft-v13-mixed",
}
BASE = "/root/autodl-tmp/models/Qwen3-4B-ms"

_engine = {}
_lock = threading.Lock()


def load_model(adapter: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_path = ADAPTERS[adapter]
    if not os.path.isdir(adapter_path):
        sys.exit(f"找不到 adapter: {adapter_path}")

    print(f"[serve] 基座 {BASE}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    print(f"[serve] 挂载 adapter {adapter} ({adapter_path})", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    _engine["tok"] = tok
    _engine["model"] = model
    _engine["adapter"] = adapter
    print("[serve] 模型就绪", flush=True)


def _render(messages):
    tok = _engine["tok"]
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tok(text, return_tensors="pt").to(_engine["model"].device)


def _gen_kwargs(req, streamer=None):
    return dict(
        max_new_tokens=int(req.get("max_tokens") or 512),
        do_sample=(req.get("temperature") or 0) > 0,
        temperature=max(float(req.get("temperature") or 0.7), 1e-5),
        top_p=float(req.get("top_p") or 0.9),
        repetition_penalty=1.1,
        pad_token_id=_engine["tok"].pad_token_id
        or _engine["tok"].eos_token_id,
        streamer=streamer,
    )


def build_app(model_name: str):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "atri",
                }
            ],
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "adapter": _engine.get("adapter")}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        import torch
        from transformers import TextIteratorStreamer

        messages = body.get("messages") or []
        if not messages:
            return JSONResponse({"error": "messages required"}, status_code=400)
        stream = bool(body.get("stream"))

        inputs = _render(messages)
        prompt_len = inputs["input_ids"].shape[1]
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not stream:
            with _lock:
                with torch.no_grad():
                    out = _engine["model"].generate(
                        **inputs, **_gen_kwargs(body)
                    )
            text = _engine["tok"].decode(
                out[0][prompt_len:], skip_special_tokens=True
            ).strip()
            return {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text or "……"},
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
            _engine["tok"], skip_prompt=True, skip_special_tokens=True
        )
        kwargs = _gen_kwargs(body, streamer=streamer)

        def run():
            with _lock:
                with torch.no_grad():
                    _engine["model"].generate(**inputs, **kwargs)

        def event_stream():
            t = threading.Thread(target=run, daemon=True)
            t.start()
            first = True
            for piece in streamer:
                if not piece:
                    continue
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": (
                                {"role": "assistant", "content": piece}
                                if first
                                else {"content": piece}
                            ),
                            "finish_reason": None,
                        }
                    ],
                }
                first = False
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="v13", choices=list(ADAPTERS))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-name", default="atri")
    args = ap.parse_args()

    load_model(args.adapter)
    import uvicorn

    uvicorn.run(
        build_app(args.model_name),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
