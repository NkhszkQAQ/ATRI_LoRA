#!/usr/bin/env python3
"""v3 数据清洗：针对 2026-11-01 日程里记录的 4 类问题

1. 「高性能」被当名词/生造复合词用 -> 定向改写
2. 「优先事项」过高频(4.1%) -> 降到 ~2%
3. 「呜哇」0 次(persona 要求) -> 从 synthetic 筛干净样本补入
4. 「别盯着」0 次 -> 数据侧无需处理（模型自创，靠 prompt 约束）
"""
import json, re, sys, argparse
from pathlib import Path

# ---- 规则 1：生造复合词 / 名词化「高性能」-> 改写 ----
COMPOUND_FIXES = [
    ("高性能实验音", "实验音"),
    ("高性能独裁", "独裁"),
    ("高性能电子眼", "电子眼"),
    ("高性能打包", "打包"),
    ("高性能闹钟", "闹钟"),
    ("高性能应答", "应答"),
    ("高性能顽固", "顽固"),
    ("高性能到场", "到场"),
    ("高性能契约", "契约"),
    ("高性能武装", "武装"),
    ("我的高性能发挥出来", "我的实力发挥出来"),
    ("高性能可不是枕头能比的", "我可不是枕头能比的"),
    ("高性能也不干", "我也不干"),
    ("必须展现出我的高性能", "必须展现出我的实力"),
    ("我的高性能AI分析", "我的分析机能"),
    ("我的高性能AI", "我的AI"),
    ("将我的高性能烙印", "将我的实力烙印"),
    ("用高性能AI拆穿", "用分析机能拆穿"),
    ("高性能实验", "实验"),
    ("超高性能", "高性能"),
]

# ---- 规则 2：「优先事项」同义替换（降频但保语义）----
PRIORITY_FIXES = [
    ("优先事项又乱了", "脑袋又乱了"),
    ("优先事项已更新", "要紧的事已更新"),
    ("优先事项更新", "要紧的事更新"),
    ("优先事项确认", "要紧的事确认"),
    ("优先事项清单", "待办清单"),
    ("优先事项", "要紧的事"),
]

def fix_text(t):
    for a, b in COMPOUND_FIXES:
        t = t.replace(a, b)
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/root/autodl-tmp/atri/data/v2/pilot/atri_v2_pilot_clean_v22.jsonl")
    ap.add_argument("--syn", default="/root/autodl-tmp/atri/data/v2/synthetic/accepted_all.jsonl")
    ap.add_argument("--out", default="/root/autodl-tmp/atri/data/v2/pilot/atri_v3_clean.jsonl")
    ap.add_argument("--priority-target", type=int, default=8, help="优先事项目标出现次数")
    ap.add_argument("--wow-count", type=int, default=18, help="补充呜哇样本条数")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    print(f"[in]  {len(rows)} 条")

    # 1) 逐条修复合词
    fixed = 0
    for r in rows:
        for m in r.get("conversations", []):
            if m.get("from") == "gpt":
                new = fix_text(m["value"])
                if new != m["value"]:
                    fixed += 1
                    m["value"] = new
    print(f"[1] 高性能生造复合词改写: {fixed} 条回复")

    # 2) 优先事项降频：只对前 N 处保留，其余替换
    occ = []
    for i, r in enumerate(rows):
        for m in r.get("conversations", []):
            if m.get("from") == "gpt" and "优先事项" in m["value"]:
                occ.append(m)
    print(f"[2] 当前含「优先事项」回复: {len(occ)} 处")
    keep = args.priority_target
    for m in occ[keep:]:
        t = m["value"]
        for a, b in PRIORITY_FIXES:
            t = t.replace(a, b)
        m["value"] = t
    print(f"    保留前 {keep} 处, 其余已同义替换")

    # 3) 从 synthetic 补「呜哇」样本（过滤污染）
    added = 0
    if Path(args.syn).exists():
        seen = set()
        for r in rows:
            seen.add(json.dumps([m["value"] for m in r["conversations"]], ensure_ascii=False))
        syn = [json.loads(l) for l in open(args.syn, encoding="utf-8") if l.strip()]
        picked = []
        for r in syn:
            txt = " ".join(m["value"] for m in r.get("conversations", []) if m.get("from") == "gpt")
            if "呜哇" not in txt:
                continue
            # 污染过滤
            if txt.count("高性能") > 1 or "优先事项" in txt:
                continue
            key = json.dumps([m["value"] for m in r["conversations"]], ensure_ascii=False)
            if key in seen:
                continue
            # 系统提示沿用 v3 风格（避免别人名残留）
            picked.append(r)
            seen.add(key)
            if len(picked) >= args.wow_count:
                break
        for r in picked:
            r = dict(r)
            r.setdefault("meta", {})
            r["meta"]["tags"] = list(set(r["meta"].get("tags", []) + ["v3_synth_wow"]))
            rows.append(r)
        added = len(picked)
    print(f"[3] 从 synthetic 补入「呜哇」样本: {added} 条")

    # 4) 输出 + 统计
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = 0; c = {"高性能": 0, "优先事项": 0, "呜哇": 0, "哼哼": 0, "嘿嘿": 0}
    for r in rows:
        for m in r.get("conversations", []):
            if m.get("from") == "gpt":
                n += 1
                for k in c: c[k] += m["value"].count(k)
    print(f"[out] {len(rows)} 条 / assistant {n} 条 -> {args.out}")
    for k, v in c.items():
        print(f"    {k}: {v} 次 ({v/n*100:.1f}%)")

if __name__ == "__main__":
    main()
