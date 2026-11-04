import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE, ADAPTER = "Qwen/Qwen3-4B", "adapter_v22"
persona = Path("persona/system_prompt.txt").read_text(encoding="utf-8")
SYSTEM = (
    persona.replace("{{user}}", "主人").replace("{{honorific}}", "")
    .replace("{{memory}}", "（暂无长期记忆）")
)

t0 = time.time()
print("[1/4] tokenizer ...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

print("[2/4] 基座（首次下载 8 GB）...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16, device_map="mps", trust_remote_code=True
)

print("[3/4] adapter v2.2 ...", flush=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print(f"[4/4] ready in {time.time()-t0:.0f}s", flush=True)

msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "你好，你是谁？"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok([text], return_tensors="pt").to("mps")

t1 = time.time()
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=120, do_sample=True,
        temperature=0.85, top_p=0.9, pad_token_id=tok.eos_token_id,
    )
ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print("ATRI:", ans, flush=True)
print(f"生成 {len(ans)} 字，耗时 {time.time()-t1:.1f}s", flush=True)
