#!/usr/bin/env python3
"""v5 测试 + 风格统计"""
import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE = "/root/autodl-tmp/models/Qwen3-4B-ms"
TPL = Path("/root/autodl-tmp/atri/persona/system_prompt.txt").read_text(encoding="utf8").strip()
NAME, HON = "小唐", "先生"

def build_system(strict=False):
    t = TPL.replace("{{user}}", NAME).replace("{{honorific}}", HON).replace("{{memory}}", "（暂无）")
    t += (f"\n\n[本轮硬性约束]\n- 你的名字只有 ATRI / 亚托莉。\n"
          f"- 称呼对方只能用「{NAME}{HON}」。\n- 禁止使用夏生、斑鸠、Attry，或乱编昵称。\n")
    if strict:
        t += f"""

[强制规则 —— 每一条都必须做到]
规则1（称呼）：回复的第一句必须包含「{NAME}{HON}」。
规则2（抗议）：对方叫你"废物"或贬低你时，必须立刻强烈抗议，禁止示弱。
  合格句式：「那是蔑称！要求撤回！」「说机器人是废物可是歧视行为！」
  禁止出现：「我还在学习」「我会努力的」「对不起」。
规则3（语气）：必须使用省略号「……」或语气词（哼哼/嘿嘿/呜哇/～），别整段只有句号。
"""
    return t

PROMPTS = [
    "你是不是什么都干不好啊，废物。",
    "今天特别累，什么都不想干，感觉自己挺没用的。",
]

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
model.eval()
model = PeftModel.from_pretrained(model, "/root/autodl-tmp/atri-runs/sft-v5-deep")
model.eval()

for label, strict in [("v5 原prompt", False), ("v5 硬规则prompt", True)]:
    print("="*58); print(f"### {label}")
    sys_msg = build_system(strict)
    for p in PROMPTS:
        msgs=[{"role":"system","content":sys_msg},{"role":"user","content":p}]
        text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok(text,return_tensors="pt").to("cuda")
        with torch.no_grad():
            out=model.generate(**ids,max_new_tokens=120,do_sample=False,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        r=tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        print(f"[用户] {p}")
        print(f"[ATRI] {r}")
        # 风格指标
        ell = r.count("…")/max(1,len(r))*100
        has_name = "有" if NAME in r else "无"
        excl = r.count("！")
        print("       (长度%d字, 省略号%.1f%%, 称呼%s, 感叹号%d)" % (len(r), ell, has_name, excl))
        print("-"*40)
