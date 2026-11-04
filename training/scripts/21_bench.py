"""21: 固定测试集跑分——对比不同基座/版本的风格指标。
用法（本机 MPS 或服务器 CUDA 都能跑）：
  python training/scripts/21_bench.py --base Qwen/Qwen3-4B --adapter models/adapter_v22 --tag qwen-v5
  python training/scripts/21_bench.py --base /root/autodl-tmp/models/GLM-4-9B-0414 --tag glm-v6
  # 服务器布局（atri/data、atri/eval）用 --root 指定：
  python scripts/21_bench.py --root /root/autodl-tmp/atri --base /root/autodl-tmp/models/Qwen3-4B-ms --tag v6-context
输出：控制台表格 + <root>/training/eval/results 或 <root>/eval/results 下的 <tag>.json
"""
import argparse, json, re, statistics, sys, time
from pathlib import Path

DEFAULT_BENCH = ("training/eval/atri_bench.jsonl", "eval/atri_bench.jsonl")


def resolve_paths(cli_root: str | None):
    """兼容两种目录布局：仓库（training/eval）与服务器（atri/eval）。"""
    if cli_root:
        root = Path(cli_root).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent
        while root != root.parent and not (root / "persona/system_prompt.txt").exists():
            root = root.parent

    bench = next((root / p for p in DEFAULT_BENCH if (root / p).exists()), None)
    if bench is None:
        raise SystemExit(f"找不到评测集，root={root}")
    persona = root / "persona/system_prompt.txt"
    outdir = root / "training/eval/results" if (root / "training").is_dir() else root / "eval/results"
    return bench, persona, outdir


def load_system(persona: Path):
    t = persona.read_text(encoding="utf-8")
    return t.replace("{{user}}", "主人").replace("{{honorific}}", "").replace("{{memory}}", "（暂无长期记忆）")


def metrics(texts):
    def body(t):
        return re.sub(r"[。！？!?…~～♪\s]", "", t)

    lens = [len(body(t)) for t in texts]
    sents = [len([s for s in re.split(r"[。！？!?\n]", t) if s.strip()]) for t in texts]
    return {
        "回复数": len(texts),
        "平均字数": round(statistics.mean(lens), 1),
        "中位字数": round(statistics.median(lens), 1),
        "平均句数": round(statistics.mean(sents), 2),
        "省略号率%": round(100 * sum("…" in t or "..." in t for t in texts) / len(texts), 1),
        "称呼率%": round(100 * sum(("先生" in t or "小姐" in t or "主人" in t) for t in texts) / len(texts), 1),
        "语气词率%": round(100 * sum(any(k in t for k in ("哼", "嘿", "呜哇", "♪")) for t in texts) / len(texts), 1),
        "螃蟹提及": sum("螃蟹" in t for t in texts),
        "复读嫌疑": sum(1 for t in texts if len(body(t)) <= 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--root", default=None, help="项目根目录；默认自动推断")
    ap.add_argument("--template", default=None, help="不填则用 tokenizer 自带 chat_template")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--load-in-4bit", action="store_true", help="4bit 加载（省显存，可与训练并行）")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    bench, persona, outdir = resolve_paths(a.root)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev == "cuda" else torch.float16
    print(f"[1/3] 加载基座 {a.base} → {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if a.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            a.base, quantization_config=bnb, device_map=dev, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, device_map=dev, trust_remote_code=True)
    if a.adapter:
        from peft import PeftModel
        print(f"[2/3] 挂 adapter {a.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    system = load_system(persona)
    items = [json.loads(l) for l in bench.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[3/3] 跑 {len(items)} 题", flush=True)
    texts, rows = [], []
    for it in items:
        msgs = [{"role": "system", "content": system}] + it["messages"]
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            prompt = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, **kw)
        inputs = tok([prompt], return_tensors="pt").to(dev)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=True,
                                 temperature=a.temperature, top_p=0.9, repetition_penalty=1.05,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        texts.append(ans)
        rows.append({"case": it["meta"]["case"], "q": it["messages"][-1]["content"], "a": ans,
                     "sec": round(time.time() - t0, 1)})
        print(f"  [{it['meta']['case']}] {it['messages'][-1]['content'][:18]} → {ans[:50]}", flush=True)

    m = metrics(texts)
    print("\n=== " + a.tag + " ===")
    for k, v in m.items():
        print(f"  {k}: {v}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.tag}.json").write_text(
        json.dumps({"tag": a.tag, "base": a.base, "adapter": a.adapter, "metrics": m, "rows": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {outdir / (a.tag + '.json')}")


if __name__ == "__main__":
    sys.exit(main())
