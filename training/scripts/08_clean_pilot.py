#!/usr/bin/env python3
"""清洗 pilot 打包集 → 可直接训练的 jsonl。

按 README 交接说明第 3 步：
1. 回复里的字面 {{user}} 换成统一假名（线上再由 system 改口）
2. system 统一换成 data/v2/SYSTEM_TEMPLATE.txt，并填实 {{memory}}
3. 输出新文件 + 在 data/dataset_info.json 注册，不动原始 pilot

用法:
  python scripts/08_clean_pilot.py
  python scripts/08_clean_pilot.py --user-name 小林 --honorific 先生
  python scripts/08_clean_pilot.py --in data/v2/pilot/atri_v2_pilot.jsonl \\
      --out data/v2/pilot/atri_v2_pilot_clean.jsonl --name atri_v2_pilot_clean
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
SYSTEM_TEMPLATE = (V2 / "SYSTEM_TEMPLATE.txt").read_text(encoding="utf-8").strip()
EMPTY_MEMORY = "（暂无）"
PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")
MEM_HEADER = re.compile(r"^\[?关于.*记忆")


def extract_memory(system: str) -> str:
    """从旧 system 尾部抠出记忆块；抠不到返回空串。"""
    lines = system.splitlines()
    start = None
    for i, line in enumerate(lines):
        if MEM_HEADER.search(line.strip()):
            start = i + 1
    if start is None:
        return ""
    bullets = [ln.strip() for ln in lines[start:] if ln.strip().startswith(("-", "•", "*"))]
    return "\n".join(bullets)


def render_system(user_name: str, honorific: str, memory: str) -> str:
    return (
        SYSTEM_TEMPLATE.replace("{{user}}", user_name)
        .replace("{{honorific}}", honorific)
        .replace("{{memory}}", memory or EMPTY_MEMORY)
    )


def clean_text(text: str, user_name: str, honorific: str) -> str:
    return text.replace("{{user}}", user_name).replace("{{honorific}}", honorific)


def ensure_dataset_info(name: str, rel: str) -> None:
    info_path = ROOT / "data" / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info[name] = {
        "file_name": rel,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/v2/pilot/atri_v2_pilot.jsonl")
    ap.add_argument("--out", dest="dst", default="data/v2/pilot/atri_v2_pilot_clean.jsonl")
    ap.add_argument("--name", default="atri_v2_pilot_clean", help="dataset_info.json 里的注册名")
    ap.add_argument("--user-name", default="小林", help="替换 {{user}} 的统一假名")
    ap.add_argument("--honorific", default="先生", help="替换 {{honorific}} 的敬称")
    args = ap.parse_args()

    src = (ROOT / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    dst = (ROOT / args.dst) if not Path(args.dst).is_absolute() else Path(args.dst)
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    out_rows = []
    mem_filled = mem_empty = replied_name = 0
    for r in rows:
        old_system = r.get("system") or ""
        memory = extract_memory(old_system)
        if memory:
            mem_filled += 1
        elif "{{memory}}" in old_system:
            mem_empty += 1
        system = render_system(args.user_name, args.honorific, memory)
        convs = []
        for c in r.get("conversations") or []:
            value = clean_text(str(c.get("value", "")), args.user_name, args.honorific)
            if c.get("from") == "gpt" and "{{user}}" in str(c.get("value", "")):
                replied_name += 1
            convs.append({"from": c.get("from"), "value": value})
        out_rows.append({"system": system, "conversations": convs, "meta": r.get("meta", {})})

    # 自检：不允许残留任何占位符
    bad = [i for i, r in enumerate(out_rows) if PLACEHOLDER.search(json.dumps(r, ensure_ascii=False))]
    if bad:
        print(f"[FAIL] 仍有占位符残留：{len(bad)} 条，idx={bad[:10]}")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        rel = dst.relative_to(ROOT / "data").as_posix()
    except ValueError:
        rel = dst.as_posix()
    ensure_dataset_info(args.name, rel)

    short = sum(
        1
        for r in out_rows
        if len("".join(c["value"] for c in r["conversations"] if c["from"] == "gpt")) < 40
    )
    multi = sum(1 for r in out_rows if len(r["conversations"]) > 2)
    print(f"wrote {dst} n={len(out_rows)}")
    print(f"memory filled={mem_filled} empty={{暂无}}={mem_empty}")
    print(f"gpt 回复含 {{{{user}}}} 已替换={replied_name}  | 剩余占位符=0")
    print(f"短样本(assistant<40字)={short}  multi_turn(>1轮)={multi}")
    print(f"registered dataset: {args.name} -> {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
