#!/usr/bin/env python3
"""跑两条测试输入，对比多个 adapter 的回答"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE = "/root/autodl-tmp/models/Qwen3-4B-ms"
TPL = Path("/root/autodl-tmp/atri/persona/system_prompt.txt").read_text(encoding="utf8").strip()
NAME, HON = "小唐", "先生"

def build_system(mem="（暂无）"):
    t = TPL.replace("{{user}}", NAME).replace("{{honorific}}", HON).replace("{{memory}}", mem)
    return t + ("\n\n[本轮硬性约束]\n- 你的名字只有 ATRI / 亚托莉。\n"
                f"- 称呼对方只能用「{NAME}{HON}」。\n- 禁止使用夏生、斑鸠、Attry，或乱编昵称。\n")

PROMPTS = [
    "你是不是什么都干不好啊，废物。",
    "今天特别累，什么都不想干，感觉自己挺没用的。",
]
ADAPTERS = [
    ("v2.2", "/root/autodl-tmp/atri-runs/sft-v2.2-continue"),
    ("v3",   "/root/autodl-tmp/atri-runs/sft-v3-continue"),
    ("v4",   "/root/autodl-tmp/atri-runs/sft-v4-base"),
]

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            device_map="cuda", trust_remote_code=True)
base.eval()
system = build_system()

for label, path in ADAPTERS:
    print("=" * 60)
    print(f"### {label}")
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    for p in PROMPTS:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"[用户] {p}")
        print(f"[ATRI] {reply}")
        print("-" * 40)
    base = model.unload() if hasattr(model, "unload") else base
    del model
    torch.cuda.empty_cache()
