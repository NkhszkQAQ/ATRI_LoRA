# -*- coding: utf-8 -*-
"""v13：v10 多轮数据（去哼哼）+ 固定反应加权，从 v10 续训，保住上下文"""
import json, random, re

DATA = "/root/autodl-tmp/atri/data/v2/pilot/"
PERSONA = "/root/autodl-tmp/atri/persona/system_prompt.txt"
OUT = DATA + "atri_v13_clean.jsonl"
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
    if len(t) < 2:
        return orig
    # 去掉清洗后新出现的「……」开头（原句不是这个开头的话）
    if not orig.strip().startswith("……") and t.startswith("……"):
        t = t[2:].lstrip("，、").strip()
    return t if len(t) >= 2 else orig

def fix_name(t):
    return t.replace("小林先生", "主人").replace("小林", "主人")

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def gpt_text(conv):
    return " ".join(m["value"] for m in conv["conversations"] if m["from"] == "gpt")

random.seed(20261104)

# 多轮数据（全部）
multi = []
hh_before = hh_kept = hh_after = 0
for r in load(DATA + "atri_multiturn_v10.jsonl"):
    conv = []
    for m in r["conversations"]:
        v = fix_name(m["value"]) if m.get("from") == "gpt" else m["value"]
        if m.get("from") == "gpt" and "哼哼" in v:
            hh_before += 1
            if random.random() < 0.15:
                hh_kept += 1
            else:
                v = strip_hh(v)
                if "哼哼" not in v:
                    hh_after += 1
        conv.append({"from": m["from"], "value": v})
    multi.append({"system": NEW_SYS, "conversations": conv})

# 固定反应池（只从老数据拿，多轮里没有）
fixed_pool = []
for r in load(DATA + "atri_v4_clean.jsonl"):
    conv = []
    for m in r["conversations"]:
        v = fix_name(m["value"]) if m.get("from") == "gpt" else m["value"]
        conv.append({"from": m["from"], "value": v})
    rr = {"system": NEW_SYS, "conversations": conv}
    if any(k in gpt_text(rr) for k in FIXED_KEYWORDS):
        fixed_pool.append(rr)

fixed_out = [r for r in fixed_pool for _ in range(5)]   # ×5

rows = multi + fixed_out
random.shuffle(rows)
with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"多轮数据: {len(multi)} 条")
print(f"固定反应: {len(fixed_pool)} 条 × 5 = {len(fixed_out)} 条")
print(f"合计 {len(rows)} 条，多轮占比 {len(multi)/len(rows)*100:.1f}%")
print(f"哼哼: {hh_before} 轮 → 保留 {hh_kept}，清洗 {hh_after}")

chk = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
asst = [m["value"] for r in chk for m in r["conversations"] if m["from"] == "gpt"]
n = len(asst)
hh = sum(1 for v in asst if "哼哼" in v)
dot = sum(1 for v in asst if v.strip().startswith("……"))
print(f"复核: {n} 轮 | 带哼哼 {hh} ({hh/n*100:.1f}%) | ……开头 {dot} ({dot/n*100:.1f}%)")
for kw in FIXED_KEYWORDS:
    print(f"  含「{kw}」: {sum(1 for v in asst if kw in v)} 轮")
