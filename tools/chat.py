#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""亚托莉 LoRA —— 手动聊天测试工具（跑在 GPU 服务器上）

用法:
    python /root/autodl-tmp/atri/chat.py                 # 默认测 v11
    python /root/autodl-tmp/atri/chat.py --adapter v5    # 换成旧版对比
    python /root/autodl-tmp/atri/chat.py --adapter v10

聊天中可用命令:
    /reset        清空当前对话，重新开始（system 不变）
    /sys          显示当前 system prompt
    /temp 0.7     改采样温度（0 = 贪心，和诊断脚本一致）
    /greedy       切成贪心解码
    /history      看当前对话轮数
    /quit         退出
"""

import argparse
import json
import os
import sys
import time

ADAPTERS = {
    "v5": "/root/autodl-tmp/atri-runs/sft-v5-deep",
    "v10": "/root/autodl-tmp/atri-runs/sft-v10-multiturn",
    "v11": "/root/autodl-tmp/atri-runs/sft-v11-mixed",
    "v12": "/root/autodl-tmp/atri-runs/sft-v12-mixed",
    "v13": "/root/autodl-tmp/atri-runs/sft-v13-mixed",
}
BASE = "/root/autodl-tmp/models/Qwen3-4B-ms"
PERSONA = "/root/autodl-tmp/atri/persona/system_prompt.txt"


def load_system_prompt():
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
    ap.add_argument("--adapter", default="v12", choices=list(ADAPTERS))
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--greedy", action="store_true", help="贪心解码（和诊断脚本一致）")
    args = ap.parse_args()

    adapter_path = ADAPTERS[args.adapter]
    if not os.path.isdir(adapter_path):
        sys.exit(f"找不到 adapter: {adapter_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"加载基座 {BASE} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    print(f"挂载 adapter: {args.adapter} ({adapter_path}) ...", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    gen = dict(
        max_new_tokens=args.max_new,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if args.greedy:
        # 贪心模式与诊断脚本严格对齐（诊断没设 repetition_penalty）
        gen.update(do_sample=False, repetition_penalty=1.0)
    else:
        gen.update(do_sample=True, temperature=args.temp, top_p=args.top_p)

    system_text = load_system_prompt()
    history = []  # [{"role": ..., "content": ...}]

    print("=" * 60)
    print(f"  ATRI 手动测试 · 版本 {args.adapter} · "
          f"{'贪心' if args.greedy else f'temp={args.temp} top_p={args.top_p}'}")
    print("  直接打字聊天；/reset 重开，/temp 0.8 调温度，/quit 退出")
    print("=" * 60, flush=True)

    while True:
        try:
            user = input("\n主人 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user:
            continue

        # ---- 内置命令 ----
        if user in ("/quit", "/exit", "/q"):
            print("再见。")
            break
        if user == "/reset":
            history = []
            print("（对话已清空）")
            continue
        if user == "/history":
            print(f"（当前 {sum(1 for m in history if m['role'] == 'user')} 轮）")
            continue
        if user == "/sys":
            print("-" * 50)
            print(system_text)
            print("-" * 50)
            continue
        if user.startswith("/temp"):
            parts = user.split()
            if len(parts) == 2:
                try:
                    t = float(parts[1])
                    if t <= 0:
                        gen.pop("temperature", None)
                        gen.pop("top_p", None)
                        gen["do_sample"] = False
                        print("（切到贪心解码）")
                    else:
                        gen.update(do_sample=True, temperature=t, top_p=args.top_p)
                        print(f"（温度 = {t}）")
                except ValueError:
                    print("用法: /temp 0.8")
            else:
                print(f"（当前温度 = {gen.get('temperature', '贪心')}）")
            continue
        if user == "/greedy":
            gen.pop("temperature", None)
            gen.pop("top_p", None)
            gen["do_sample"] = False
            print("（切到贪心解码）")
            continue

        # ---- 正常对话 ----
        history.append({"role": "user", "content": user})
        msgs = [{"role": "system", "content": system_text}] + history
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, **gen)
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        elapsed = time.time() - t0

        history.append({"role": "assistant", "content": ans})
        print(f"\nATRI  > {ans}")
        print(f"        （{elapsed:.1f}s / {n_new} tok）", flush=True)

    # 退出时把最后一段对话存下来，方便回看
    if history:
        dump = f"/root/autodl-tmp/atri/eval/chat_{args.adapter}.json"
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        with open(dump, "w", encoding="utf-8") as f:
            json.dump({"adapter": args.adapter, "turns": history}, f, ensure_ascii=False, indent=1)
        print(f"（对话已存到 {dump}）")


if __name__ == "__main__":
    main()
