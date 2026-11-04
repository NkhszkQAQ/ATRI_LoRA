"""本机挂载 SFT LoRA 的推理服务（OpenAI + Ollama 兼容流）。
用法（需 .venv-train，已装 torch+cu128 / peft / transformers / bitsandbytes）:
  .\\.venv-train\\Scripts\\python scripts\\serve_lora.py
默认: http://127.0.0.1:8008
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
from typing import Any, Iterator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
SPECIAL_RE = re.compile(r"<\|[^>]+?\|>")


def _clean_text(s: str) -> str:
    s = SPECIAL_RE.sub("", s).replace("<|im_end|>", "").replace("<|endoftext|>", "")
    s = re.sub(r"\{\{[^}]{0,40}\}\}", "", s)  # 去掉 {{笑}} 等舞台指令
    return s


class ServeState:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_id = "atri-qwen3-4b-lora"
        self.lock = threading.Lock()


STATE = ServeState()
app = FastAPI(title="Atri LoRA Serve")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(base: str, adapter: str, fourbit: bool) -> None:
    print(f"[lora-serve] loading base={base} adapter={adapter} 4bit={fourbit}")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if fourbit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    STATE.tokenizer = tok
    STATE.model = model
    print("[lora-serve] ready")


def _apply_template(messages: list[dict[str, str]], enable_thinking: bool) -> str:
    tok = STATE.tokenizer
    assert tok is not None
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # older template without enable_thinking
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _raw_token_stream(
    messages: list[dict[str, str]],
    *,
    enable_thinking_template: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    assistant_prefix: str = "",
) -> Iterator[str]:
    """Yield cleaned text pieces (special tokens stripped)."""
    prompt = _apply_template(messages, enable_thinking_template)
    if enable_thinking_template and f"{THINK_OPEN}\n\n{THINK_CLOSE}" in prompt:
        prompt = prompt.replace(f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n", "")
    if assistant_prefix:
        # 续写前缀：强制先喊对名字，再往后生成
        prompt = prompt + assistant_prefix
        yield assistant_prefix
    tok = STATE.tokenizer
    assert STATE.model is not None and tok is not None
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(STATE.model.device) for k, v in inputs.items()}
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
        repetition_penalty=1.12,
        no_repeat_ngram_size=4,
    )
    # 仅轻量下限，避免未写完就 EOS；过大下限会逼出「再见再见」死循环
    if min_new_tokens and min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = min(min_new_tokens, max_new_tokens)
    thread = threading.Thread(target=STATE.model.generate, kwargs=gen_kwargs)
    thread.start()
    buf = ""
    keep = 32
    for piece in streamer:
        buf += piece
        if len(buf) > keep * 2:
            emit, buf = buf[:-keep], buf[-keep:]
            clean = _clean_text(emit)
            if clean:
                yield clean
    if buf:
        clean = _clean_text(buf)
        if clean:
            yield clean
    thread.join()


def _parse_think_stream(text_iter: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Split Qwen <think>…</think> tags from a text stream into reasoning/content."""
    buf = ""
    in_think = False
    keep = 32
    for piece in text_iter:
        buf += piece
        while True:
            if not in_think:
                i = buf.find(THINK_OPEN)
                if i >= 0:
                    before = buf[:i]
                    if before:
                        yield {"content": before, "reasoning": ""}
                    buf = buf[i + len(THINK_OPEN) :]
                    in_think = True
                    continue
                if len(buf) > keep * 2:
                    emit, buf = buf[:-keep], buf[-keep:]
                    if emit:
                        yield {"content": emit, "reasoning": ""}
                break
            j = buf.find(THINK_CLOSE)
            if j >= 0:
                reason = buf[:j]
                if reason:
                    yield {"content": "", "reasoning": reason}
                buf = buf[j + len(THINK_CLOSE) :]
                in_think = False
                continue
            if len(buf) > keep * 2:
                emit, buf = buf[:-keep], buf[-keep:]
                if emit:
                    yield {"content": "", "reasoning": emit}
            break
    if buf:
        if in_think:
            yield {"content": "", "reasoning": buf}
        else:
            yield {"content": buf, "reasoning": ""}


def generate_stream(
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Iterator[dict[str, Any]]:
    if STATE.model is None or STATE.tokenizer is None:
        raise RuntimeError("model not loaded")
    msgs = list(messages)

    # SFT 很少自发打 <think> 标签：开启深度思考时先写较长内心独白，再生成正式回复
    if enable_thinking:
        last_user = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                last_user = m.get("content") or ""
                break
        sys0 = ""
        for m in msgs:
            if m.get("role") == "system":
                sys0 = m.get("content") or ""
                break
        # 从 system 里抠「标准称呼」提示，避免独白/回复乱叫名
        address_hint = ""
        m_addr = re.search(r"标准称呼：([^\n]+)", sys0)
        if m_addr:
            address_hint = m_addr.group(1).strip()
        if not address_hint:
            m_addr = re.search(
                r"称呼对方(?:时必须使用|只能用)「([^」]+)」",
                sys0,
            )
            if m_addr:
                address_hint = m_addr.group(1).strip()
        name_line = (
            f"对方应称呼为「{address_hint}」。你自称 ATRI/亚托莉，禁止 Attry、夏生、斑鸠等错名。\n"
            if address_hint
            else "你自称 ATRI/亚托莉；称呼对方必须与系统提示中的标准称呼一致，禁止乱改名。\n"
        )
        think_budget = min(320, max(200, max_new_tokens // 3))
        think_msgs = [m for m in msgs if m.get("role") == "system"][:1]
        think_msgs.append(
            {
                "role": "user",
                "content": (
                    f"{name_line}"
                    f"用户说：{last_user}\n\n"
                    "请写内心独白，大约 6～10 句（可分行），覆盖：\n"
                    "对方在说什么；你的心情；准备怎么回；核对称呼与自称（ATRI）。\n"
                    "不要对用户说话，不要重复同一句，不要写「再见」堆叠，不要用括号动作，写完即可停。"
                ),
            }
        )
        reason_acc: list[str] = []
        for piece in _raw_token_stream(
            think_msgs,
            enable_thinking_template=False,
            temperature=max(0.55, temperature * 0.8),
            top_p=min(0.9, top_p),
            max_new_tokens=think_budget,
            min_new_tokens=min(72, think_budget // 3),
        ):
            reason_acc.append(piece)
            yield {"content": "", "reasoning": piece}
        reason_text = "".join(reason_acc).strip()
        if reason_text.count("再见") >= 3:
            reason_text = reason_text.split("再见")[0].strip() or reason_text[:200]
        answer_msgs = [dict(m) for m in msgs]
        head = ""
        if address_hint:
            head = (
                f"【名称锁定】你=ATRI/亚托莉（禁止 Attry）。"
                f"对方={address_hint}（回复里至少称呼一次）。\n"
            )
        if answer_msgs and answer_msgs[0].get("role") == "system":
            tail = f"\n（内心独白勿复述：{reason_text[:500]}）" if reason_text else ""
            answer_msgs[0]["content"] = head + (answer_msgs[0].get("content") or "") + tail
        else:
            answer_msgs = [{"role": "system", "content": head + (reason_text[:500] or "")}] + answer_msgs
        prefix = f"{address_hint}，" if address_hint else ""
        for piece in _raw_token_stream(
            answer_msgs,
            enable_thinking_template=False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=min(max_new_tokens, 320),
            assistant_prefix=prefix,
        ):
            yield {"content": piece, "reasoning": ""}
        yield {"content": "", "reasoning": "", "done": True}
        return

    # 关闭思考：单次生成，仍兼容模型自发 <think> 标签；尽量带上称呼前缀
    sys0 = next((m.get("content") or "" for m in msgs if m.get("role") == "system"), "")
    address_hint = ""
    m_addr = re.search(r"标准称呼：([^\n]+)", sys0) or re.search(
        r"称呼对方(?:时必须使用|只能用)「([^」]+)」",
        sys0,
    )
    if m_addr:
        address_hint = m_addr.group(1).strip()
    prefix = f"{address_hint}，" if address_hint else ""

    def _iter():
        for piece in _raw_token_stream(
            msgs,
            enable_thinking_template=False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            assistant_prefix=prefix,
        ):
            yield piece

    yield from _parse_think_stream(_iter())
    yield {"content": "", "reasoning": "", "done": True}


class ChatBody(BaseModel):
    model: str = "atri-qwen3-4b-lora"
    messages: list[dict[str, str]]
    stream: bool = True
    think: bool | None = None
    enable_thinking: bool | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


@app.get("/api/tags")
def tags():
    return {"models": [{"name": STATE.model_id, "model": STATE.model_id}]}


@app.get("/health")
def health():
    return {"ok": True, "model": STATE.model_id, "loaded": STATE.model is not None}


@app.post("/api/chat")
def ollama_chat(body: ChatBody):
    if STATE.model is None:
        raise HTTPException(503, "model loading")
    enable = True
    if body.think is not None:
        enable = bool(body.think)
    elif body.enable_thinking is not None:
        enable = bool(body.enable_thinking)
    temp = body.temperature if body.temperature is not None else float(body.options.get("temperature", 0.85))
    top_p = body.top_p if body.top_p is not None else float(body.options.get("top_p", 0.9))
    max_new = body.max_tokens or int(body.options.get("num_predict", 1024))

    if not body.stream:
        content, reasoning = [], []
        with STATE.lock:
            for ev in generate_stream(
                body.messages, enable_thinking=enable, temperature=temp, top_p=top_p, max_new_tokens=max_new
            ):
                if ev.get("content"):
                    content.append(ev["content"])
                if ev.get("reasoning"):
                    reasoning.append(ev["reasoning"])
        return {
            "model": STATE.model_id,
            "message": {"role": "assistant", "content": "".join(content), "reasoning": "".join(reasoning)},
            "done": True,
        }

    def gen():
        with STATE.lock:
            for ev in generate_stream(
                body.messages, enable_thinking=enable, temperature=temp, top_p=top_p, max_new_tokens=max_new
            ):
                done = bool(ev.get("done"))
                payload = {
                    "model": STATE.model_id,
                    "message": {
                        "role": "assistant",
                        "content": ev.get("content") or "",
                        "reasoning": ev.get("reasoning") or "",
                    },
                    "done": done,
                }
                yield json.dumps(payload, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/v1/chat/completions")
def openai_chat(body: dict[str, Any]):
    """Minimal OpenAI-compatible wrapper."""
    messages = body.get("messages") or []
    enable = bool(body.get("enable_thinking", body.get("think", True)))
    stream = bool(body.get("stream", False))
    temp = float(body.get("temperature", 0.85))
    top_p = float(body.get("top_p", 0.9))
    max_new = int(body.get("max_tokens", 1024))
    chat = ChatBody(
        messages=messages,
        stream=stream,
        enable_thinking=enable,
        temperature=temp,
        top_p=top_p,
        max_tokens=max_new,
    )
    if stream:
        def gen():
            with STATE.lock:
                for ev in generate_stream(
                    messages, enable_thinking=enable, temperature=temp, top_p=top_p, max_new_tokens=max_new
                ):
                    delta = {}
                    if ev.get("content"):
                        delta["content"] = ev["content"]
                    if ev.get("reasoning"):
                        delta["reasoning"] = ev["reasoning"]
                    chunk = {
                        "id": "atri",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if ev.get("done") else None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    return ollama_chat(chat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--base", default=os.environ.get("LORA_BASE", "Qwen/Qwen3-4B"))
    ap.add_argument(
        "--adapter",
        default=os.environ.get("LORA_ADAPTER", "runs/atri-companion-qwen3-4b-sft-v0.1"),
    )
    ap.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args()
    load_model(a.base, a.adapter, fourbit=not a.no_4bit)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
