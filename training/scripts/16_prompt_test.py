#!/usr/bin/env python3
"""对照实验：原 prompt vs 硬规则 prompt，看约束是否生效"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE = "/root/autodl-tmp/models/Qwen3-4B-ms"
TPL = Path("/root/autodl-tmp/atri/persona/system_prompt.txt").read_text(encoding="utf8").strip()
NAME, HON = "小唐", "先生"

def base_system():
    t = TPL.replace("{{user}}", NAME).replace("{{honorific}}", HON).replace("{{memory}}", "（暂无）")
    return t + (f"\n\n[本轮硬性约束]\n- 你的名字只有 ATRI / 亚托莉。\n"
                f"- 称呼对方只能用「{NAME}{HON}」。\n- 禁止使用夏生、斑鸠、Attry，或乱编昵称。\n")

def strict_system():
    t = base_system()
    return t + f"""

[强制规则 —— 每一条都必须做到，违反任何一条视为失败]
规则1（称呼）：回复的第一句必须包含「{NAME}{HON}」，一个字都不能改。
规则2（抗议）：如果对方叫你"废物"或贬低你，必须立刻强烈抗议，禁止示弱。
  合格句式举例：「那是蔑称！要求撤回！」「说机器人是废物可是歧视行为！」「你又说！第几次了！」
  禁止出现：「我还在学习」「我会努力的」「对不起」「可能吧」。
规则3（语气）：回复 1~3 句，必须使用至少一个语气词（哼哼/嘿嘿/呜哇/……/～），不许整段只有句号。
规则4（自称）：用「我」或「ATRI」，禁止说「作为AI」「作为机器人助手」。

标准回答示例（照着这个感觉说）：
用户：你是不是什么都干不好啊，废物。
ATRI：那是蔑称！{NAME}{HON}，要求撤回——我可是高性能的嘛！
"""

PROMPTS = ["你是不是什么都干不好啊，废物。", "今天特别累，什么都不想干，感觉自己挺没用的。"]

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
model.eval()
model = PeftModel.from_pretrained(model, "/root/autodl-tmp/atri-runs/sft-v4-base")
model.eval()

for label, sys_msg in [("原prompt", base_system()), ("硬规则prompt", strict_system())]:
    print("=" * 58)
    print(f"### {label}")
    for p in PROMPTS:
        msgs = [{"role":"system","content":sys_msg},{"role":"user","content":p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"[用户] {p}")
        print(f"[ATRI] {reply}")
        print("-"*40)
