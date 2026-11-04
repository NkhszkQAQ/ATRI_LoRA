import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "/root/autodl-tmp/models/Qwen3-4B-ms"
ADAPTER = "/root/autodl-tmp/atri-runs/sft-v13-mixed"
EVAL = "/root/autodl-tmp/atri/eval/v5_diag.jsonl"
PERSONA = "/root/autodl-tmp/atri/persona/system_prompt.txt"
OUT = "/root/autodl-tmp/atri/eval/results/v13_diag_result.json"

sys_text = open(PERSONA, encoding='utf-8').read()
for k, v in {"{{user}}": "主人", "{{honorific}}": "", "{{memory}}": "（暂无长期记忆）"}.items():
    sys_text = sys_text.replace(k, v)

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

rows = [json.loads(l) for l in open(EVAL, encoding='utf-8') if l.strip()]
print(f"题数: {len(rows)}", flush=True)
results = []
for i, r in enumerate(rows):
    user = r['messages'][0]['content']
    msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    meta = r.get('meta', {})
    results.append({"i": i+1, "category": meta.get('category',''), "user": user, "answer": ans, "expect": meta.get('expect',''), "elapsed": round(time.time()-t0,1)})
    print(f"[{i+1}/{len(rows)}] {meta.get('category','')} | {ans[:80]}", flush=True)

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(results, open(OUT,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
print("已保存:", OUT, flush=True)
