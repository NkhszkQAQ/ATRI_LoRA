"""构建 Qwen3-8B 角色底座数据：1448 条原作台词 + 921 条 Claude 生成多轮（= atri_v11_mixed_r 的合并集）

清洗：
1. system 全部换成统一新模板（原数据有 11 种旧模板，称呼是「小林先生」「小苏先生」等）
2. gpt 文本里「小X先生」「小林」→「主人」
3. 哼哼降频 85%（v10 多轮里 87.6% 带哼哼，v11 训出来被用户否过）

用法（服务器上）： /root/autodl-tmp/atri-env/bin/python build_8b_base.py
"""
import json, random, re

DATA_R = "/root/autodl-tmp/atri/data_r/"
PERSONA = "/root/autodl-tmp/atri/persona/system_prompt.txt"
SRC = DATA_R + "atri_v11_mixed_r.jsonl"
OUT = DATA_R + "atri_8b_base.jsonl"

t = open(PERSONA, encoding="utf-8").read()
for k, v in {"{{user}}": "主人", "{{honorific}}": "", "{{memory}}": "（暂无长期记忆）"}.items():
    t = t.replace(k, v)
NEW_SYS = t

NAME_RE = re.compile(r"小[^\s，。！？、…]{1,2}先生|小林")


def fix_name(s):
    return NAME_RE.sub("主人", s)


def strip_hh(t):
    orig = t
    for a, b in [
        ("……哼哼，", "……"), ("……哼哼。", "……。"), ("……哼哼！", "……！"),
        ("，哼哼！", "！"), ("，哼哼。", "。"), ("，哼哼", ""),
        ("哼哼，", ""), ("哼哼。", ""), ("哼哼！", ""), ("哼哼……", "……"),
        ("哼哼", ""),
    ]:
        t = t.replace(a, b)
    t = re.sub(r"[，、]+([。！？…])", r"\1", t)
    t = re.sub(r"^[，、。\s]+", "", t)
    t = t.strip()
    return orig if len(t) < 2 else t


random.seed(20261104)
rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
out = []
hh_hit = hh_drop = name_fix = 0
for r in rows:
    convs = []
    for m in r["conversations"]:
        v = m["value"]
        if m["from"] == "gpt":
            nv = fix_name(v)
            if nv != v:
                name_fix += 1
            if "哼哼" in nv:
                hh_hit += 1
                if random.random() < 0.85:
                    nv = strip_hh(nv)
                    hh_drop += 1
            v = nv
        convs.append({"from": m["from"], "value": v})
    out.append({"system": NEW_SYS, "conversations": convs})

with open(OUT, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"输出 {len(out)} 条 → {OUT}")
print(f"称呼修正 {name_fix} 处 | 哼哼命中 {hh_hit} | 删除 {hh_drop}")
# 统计
import collections
first = collections.Counter()
for r in out:
    g = [m["value"] for m in r["conversations"] if m["from"] == "gpt"]
    first[g[0][:6]] += 1
print("gpt 首句 top5:", first.most_common(5))
