"""本地运行 ATRI（Mac / MPS）。用法：python run_local.py [adapter目录]"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-4B"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "adapter_v22"

# ---- system prompt ----
persona = Path("persona/system_prompt.txt").read_text(encoding="utf-8")
SYSTEM = (
    persona.replace("{{user}}", "主人")
    .replace("{{honorific}}", "")
    .replace("{{memory}}", "（暂无长期记忆）")
) + "\n\n[本轮硬性约束]\n- 你的名字只有 ATRI / 亚托莉。\n- 称呼对方只能用「主人」。\n"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = pick_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"[1/3] 设备 {device}，精度 {dtype}")

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    print(f"[2/3] 加载基座 {BASE} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dtype, device_map=device, trust_remote_code=True
    )
    print(f"[3/3] 挂载 adapter {ADAPTER} ...")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print("就绪。输入 exit 退出。\n")

    history = []
    while True:
        try:
            user = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in {"exit", "quit", "q"}:
            break

        msgs = [{"role": "system", "content": SYSTEM}]
        for u, a in history:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})

        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tok([text], return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.85,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tok.eos_token_id,
            )
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"ATRI：{reply}\n")
        history.append((user, reply))


if __name__ == "__main__":
    main()
