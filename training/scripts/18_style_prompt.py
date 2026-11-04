#!/usr/bin/env python3
"""few-shot 注入原作句式，看能否把省略号率拉回来"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE = "/root/autodl-tmp/models/Qwen3-4B-ms"
TPL = Path("/root/autodl-tmp/atri/persona/system_prompt.txt").read_text(encoding="utf8").strip()
NAME, HON = "小唐", "先生"

BASE_SYS = TPL.replace("{{user}}", NAME).replace("{{honorific}}", HON).replace("{{memory}}", "（暂无）") + \
    f"\n\n[本轮硬性约束]\n- 你的名字只有 ATRI / 亚托莉。\n- 称呼对方只能用「{NAME}{HON}」。\n- 禁止使用夏生、斑鸠、Attry，或乱编昵称。\n"

STYLE_SYS = BASE_SYS + f"""

[你的说话方式 —— 必须模仿这些语感]
ATRI 说话的特征：句子短，大量使用省略号「……」表示停顿/犹豫，情绪上来会喊出来。真实例子：
- 「……那个，{NAME}{HON}。」
- 「根据我的高性能AI分析，{NAME}{HON}这是在扯开话题」
- 「……废物是对机器人的蔑称。要求撤回」
- 「呜哇……我、我才没有！」
- 「哼哼，毕竟我是高性能的嘛」
- 「……我还在学习呢。真的。」
- 「啊！你又说废物！听好了{NAME}{HON}，说别人废物的人才是废物！」

注意：省略号是你说话的一部分，不是可有可无的装饰。几乎每句话都该有停顿感。
"""

PROMPTS = [
    "你是不是什么都干不好啊，废物。",
    "今天特别累，什么都不想干，感觉自己挺没用的。",
    "亚托莉，你在干嘛呢？",
]

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
model.eval()
model = PeftModel.from_pretrained(model, "/root/autodl-tmp/atri-runs/sft-v5-deep")
model.eval()

for label, sys_msg in [("v5 + 基础prompt", BASE_SYS), ("v5 + 句式few-shot", STYLE_SYS)]:
    print("="*56); print("###", label)
    for p in PROMPTS:
        msgs=[{"role":"system","content":sys_msg},{"role":"user","content":p}]
        text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok(text,return_tensors="pt").to("cuda")
        with torch.no_grad():
            out=model.generate(**ids,max_new_tokens=100,do_sample=False,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        r=tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        ell = r.count("…")/max(1,len(r))*100
        has = "有" if NAME in r else "无"
        print("[用户] %s" % p)
        print("[ATRI] %s" % r)
        print("       (长度%d, 省略号%.0f%%, 称呼%s)" % (len(r), ell, has))
        print("-"*38)
