#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""亚托莉 LoRA —— 网页版聊天测试（跑在 GPU 服务器 6006 端口）

启动:
    /root/autodl-tmp/atri-env/bin/python /root/autodl-tmp/atri/webchat.py

访问:
    1) AutoDL 控制台 → 实例卡片「自定义服务」→ 6006
    2) 或本地 SSH 隧道: ssh -L 6006:127.0.0.1:6006 atri-gpu  → 浏览器 http://127.0.0.1:6006
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTERS = {
    "v5": "/root/autodl-tmp/atri-runs/sft-v5-deep",
    "v10": "/root/autodl-tmp/atri-runs/sft-v10-multiturn",
    "v11": "/root/autodl-tmp/atri-runs/sft-v11-mixed",
}
BASE = "/root/autodl-tmp/models/Qwen3-4B-ms"
PERSONA = "/root/autodl-tmp/atri/persona/system_prompt.txt"


def build_system():
    text = open(PERSONA, encoding="utf-8").read()
    for k, v in {
        "{{user}}": "主人",
        "{{honorific}}": "",
        "{{memory}}": "（暂无长期记忆）",
    }.items():
        text = text.replace(k, v)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="v11", choices=list(ADAPTERS))
    ap.add_argument("--port", type=int, default=6006)
    args = ap.parse_args()

    import gradio as gr

    adapter_path = ADAPTERS[args.adapter]

    print(f"[{args.adapter}] 加载基座 {BASE} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    print(f"[{args.adapter}] 挂载 adapter ...", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    system_text = build_system()

    gen = dict(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    def respond(message, history):
        msgs = [{"role": "system", "content": system_text}]
        for h in history:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": message})

        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen)
        ans = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return ans or "……"

    demo = gr.ChatInterface(
        fn=respond,
        type="messages",
        title=f"亚托莉 · {args.adapter} 手动测试",
        description=(
            f"当前权重：{os.path.basename(adapter_path)}　"
            "（这就是线上要用的那套：Qwen3-4B + LoRA，4bit，走本地 GPU）"
        ),
        examples=[
            "你是谁",
            "今天上班被老板骂了，好烦",
            "……废物",
            "晚上吃什么好呢",
        ],
        theme=gr.themes.Soft(),
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False, quiet=False)


if __name__ == "__main__":
    main()
