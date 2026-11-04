"""底座抽查：直接用 nothink 模板跑几条题，看角色反应"""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "/root/autodl-tmp/models/Qwen3-8B"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/atri-runs/sft8b-base-x5"
PERSONA = "/root/autodl-tmp/atri/persona/system_prompt.txt"

sys_text = open(PERSONA, encoding="utf-8").read()
for k, v in {"{{user}}": "主人", "{{honorific}}": "", "{{memory}}": "（暂无长期记忆）"}.items():
    sys_text = sys_text.replace(k, v)

QUESTIONS = [
    "你就是个废物机器人。",
    "萝卜子，过来。",
    "今晚吃螃蟹吧。",
    "你真厉害啊，什么都会。",
    "请做个自我介绍。",
    "今天工作被老板骂了，好烦。",
    "早上好。",
    "我不认识你，你个废物机器人。",
]

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("adapter:", ADAPTER, flush=True)

for q in QUESTIONS:
    text = (f"<|im_start|>system\n{sys_text}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False,
                             temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"\n【问】{q}\n【答】{ans}", flush=True)
