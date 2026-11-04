"""GLM-4-9B 就绪检查：模型文件、tokenizer、LLaMA-Factory 模板、训练配置。
用法（服务器上）：
  /root/autodl-tmp/atri-env/bin/python scripts/check_glm_ready.py
"""
import json
import os
import sys

BASE = "/root/autodl-tmp/models/GLM-4-9B-0414"
ATRI = "/root/autodl-tmp/atri"
CONFIGS = ["sft_glm4_9b_qlora.yaml", "sft_glm4_9b_qlora_16g.yaml", "sft_glm4_9b_qlora_windows.yaml"]
ok = True


def check(name, cond, detail=""):
    global ok
    mark = "✓" if cond else "✗"
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


print("[1/4] 模型文件")
check("目录存在", os.path.isdir(BASE))
total = 0
if os.path.isdir(BASE):
    fs = sorted(os.listdir(BASE))
    total = sum(os.path.getsize(os.path.join(BASE, f)) for f in fs if os.path.isfile(os.path.join(BASE, f)))
    print(f"    文件 {len(fs)} 个，共 {total/1024**3:.2f} G")
    check("config.json", os.path.exists(f"{BASE}/config.json"))
    check("tokenizer 文件", os.path.exists(f"{BASE}/tokenizer_config.json"))
    shards = [f for f in fs if f.endswith(".safetensors")]
    check("权重分片 >= 4", len(shards) >= 4, f"实际 {len(shards)} 个")
    check("总大小 >= 17G", total >= 17 * 1024**3)

print("\n[2/4] AutoConfig / tokenizer")
try:
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
    check("AutoConfig", True, f"model_type={cfg.model_type}")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    check("AutoTokenizer", True, f"vocab={tok.vocab_size}")
except Exception as e:
    check("加载 config/tokenizer", False, repr(e)[:120])

print("\n[3/4] glm4 模板渲染")
try:
    from llamafactory.data.template import TEMPLATES, get_template_and_fix_tokenizer
    check("LLaMA-Factory 支持 glm4", "glm4" in TEMPLATES)
    row = json.loads(open(f"{ATRI}/data/v2/pilot/atri_windows_v6.jsonl", encoding="utf-8").readline())
    tpl = TEMPLATES["glm4"]
    rendered = tpl.format_message("user", "你好呀") if hasattr(tpl, "format_message") else None
    check("模板对象可用", True, str(type(tpl).__name__))
except Exception as e:
    check("模板渲染", False, repr(e)[:120])

print("\n[4/4] 训练配置")
for c in CONFIGS:
    p = f"{ATRI}/configs/{c}"
    if not os.path.exists(p):
        check(c, False, "文件不存在")
        continue
    import yaml
    d = yaml.safe_load(open(p, encoding="utf-8"))
    di = json.load(open(f"{ATRI}/data/dataset_info.json", encoding="utf-8"))
    ds = d["dataset"]
    exists = ds in di
    n = sum(1 for _ in open(f"{ATRI}/data/{di[ds]['file_name']}", encoding="utf-8")) if exists else 0
    check(c, d["model_name_or_path"] == BASE and exists and n > 0,
          f"dataset={ds}({n}行) template={d['template']} bs={d['per_device_train_batch_size']}")

print("\n结论:", "GLM 就绪，有卡即可开训" if ok else "有缺失项，见上面 ✗")
sys.exit(0 if ok else 1)
