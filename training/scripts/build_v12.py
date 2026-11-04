# -*- coding: utf-8 -*-
"""v12 训练数据：多轮当主体 + 固定反应加权 + 哼哼降频 + 称呼统一"""
import json, random, re

DATA = "/root/autodl-tmp/atri/data/v2/pilot/"
PERSONA = "/root/autodl-tmp/atri/persona/system_prompt.txt"
OUT = DATA + "atri_v12_clean.jsonl"

FIXED_KEYWORDS = ["废物", "萝卜子", "蔑称", "螃蟹"]

def build_system():
    t = open(PERSONA, encoding="utf-8").read()
    for k, v in {"{{user}}": "主人", "{{honorific}}": "", "{{memory}}": "（暂无长期记忆）"}.items():
        t = t.replace(k, v)
    return t

NEW_SYS = build_system()

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

def fix_name(t):
    return t.replace("小林先生", "主人").replace("小林", "主人")

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def conv_clean(r):
    out = []
    for m in r["conversations"]:
        v = fix_name(m["value"]) if m.get("from") == "gpt" else m["value"]
        out.append({"from": m["from"], "value": v})
    return {"system": NEW_SYS, "conversations": out}

def gpt_text(conv):
    return " ".join(m["value"] for m in conv["conversations"] if m["from"] == "gpt")

random.seed(20261104)

# ---------- 1) 多轮数据 ----------
multi = []
hh_before = hh_after = hh_kept = 0
for r in load(DATA + "atri_multiturn_v10.jsonl"):
    rr = conv_clean(r)
    for m in rr["conversations"]:
        if m["from"] == "gpt" and "哼哼" in m["value"]:
            hh_before += 1
            if random.random() < 0.15:
                hh_kept += 1
            else:
                m["value"] = strip_hh(m["value"])
                if "哼哼" not in m["value"]:
                    hh_after += 1
    multi.append(rr)

# ---------- 2) 老数据分池（只看她自己的话，不看 system） ----------
old = []
for r in load(DATA + "atri_v4_clean.jsonl"):
    rr = conv_clean(r)
    txt = gpt_text(rr)
    old.append(("fixed" if any(k in txt for k in FIXED_KEYWORDS) else "normal", rr))

fixed_pool = [r for t, r in old if t == "fixed"]
normal_pool = [r for t, r in old if t == "normal"]

fixed_out = []
for r in fixed_pool:
    fixed_out.extend([r] * 5)
normal_out = random.sample(normal_pool, min(300, len(normal_pool)))

rows = multi + fixed_out + normal_out
random.shuffle(rows)

with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"多轮数据:   {len(multi)} 条")
print(f"固定反应池: {len(fixed_pool)} 条 → 复制 5 份 = {len(fixed_out)} 条")
print(f"普通单轮:   {len(normal_out)} 条（从 {len(normal_pool)} 条里抽）")
print(f"合计:       {len(rows)} 条，多轮占比 {len(multi)/len(rows)*100:.1f}%")
print(f"v10 带哼哼回复 {hh_before} 轮 → 保留 {hh_kept}，清洗 {hh_after}")

chk = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
asst = sum(1 for r in chk for m in r["conversations"] if m["from"] == "gpt")
hh = sum(1 for r in chk for m in r["conversations"] if m["from"] == "gpt" and "哼哼" in m["value"])
print(f"复核：她的回复 {asst} 轮，带哼哼 {hh} 轮（{hh/asst*100:.1f}%）")
for kw in FIXED_KEYWORDS:
    n = sum(1 for r in chk if kw in gpt_text(r))
    print(f"复核：含「{kw}」样本 {n} 条")
bad = sum(1 for r in chk if "小林" in json.dumps(r, ensure_ascii=False))
print(f"复核：「小林」残留 {bad} 条")
