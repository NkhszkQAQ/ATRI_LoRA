"""底座抽查第二组：重点看「回答和输入有没有关系」"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "/root/autodl-tmp/models/Qwen3-8B"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/atri-runs/sft8b-base-x5"
PERSONA = "/root/autodl-tmp/atri/persona/system_prompt.txt"

sys_text = open(PERSONA, encoding="utf-8").read()
for k, v in {"{{user}}": "主人", "{{honorific}}": "", "{{memory}}": "（暂无长期记忆）"}.items():
    sys_text = sys_text.replace(k, v)

QUESTIONS = [
    "我手机屏幕碎了，是修还是换新的？",
    "你觉得我该不该辞职？",
    "今天地铁上有人踩我脚，还不道歉。",
    "你知道什么是区块链吗？",
    "我好累，不想上班。",
    "陪我说说话。",
    "我养了一只猫，叫团子。",
    "晚上吃火锅还是烧烤？",
    "你是不是在假装理解我？",
    "算了，不聊了，我睡了。",
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
        out = model.generate(**inputs, max_new_tokens=180, do_sample=False,
                             temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"\n【问】{q}\n【答】{ans}", flush=True)
