"""thinking 版诊断：用 LLaMA-Factory qwen3 模板的裸格式构造 prompt（不带 Qwen3 原生 thinking 提示），
让模型自己生成 <think> 块，然后拆出「内心独白」和「正式回复」。

用法：
  python think_diag.py <adapter_dir> <out.json> [--thinking]
"""
import json, os, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = os.environ.get("BASE_MODEL", "/root/autodl-tmp/models/Qwen3-4B-ms")
EVAL = "/root/autodl-tmp/atri/eval/v5_diag.jsonl"
PERSONA = "/root/autodl-tmp/atri/persona/system_prompt.txt"

adapter = sys.argv[1]
out_path = sys.argv[2]
max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 400

sys_text = open(PERSONA, encoding="utf-8").read()
for k, v in {"{{user}}": "主人", "{{honorific}}": "", "{{memory}}": "（暂无长期记忆）"}.items():
    sys_text = sys_text.replace(k, v)

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model.eval()


def build_prompt(user):
    # 同 llamafactory qwen3 模板：system / user / assistant 三个 im_start 段，assistant 后不加任何提示
    return (f"<|im_start|>system\n{sys_text}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def split_think(text):
    m = re.search(r"<think>(.*?)</think>\s*(.*)", text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


rows = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
print(f"题数: {len(rows)} | adapter: {adapter}", flush=True)
results = []
for i, r in enumerate(rows):
    user = r["messages"][0]["content"]
    text = build_prompt(user)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    think, reply = split_think(ans)
    meta = r.get("meta", {})
    results.append({"i": i + 1, "category": meta.get("category", ""), "user": user,
                    "think": think, "answer": reply, "raw": ans,
                    "thinking_len": len(think), "elapsed": round(time.time() - t0, 1)})
    print(f"[{i+1}/{len(rows)}] {meta.get('category','')} | think={len(think)}字 | {reply[:60]}", flush=True)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n_think = sum(1 for x in results if x["think"])
print(f"已保存: {out_path} | 带思考 {n_think}/{len(results)}", flush=True)
