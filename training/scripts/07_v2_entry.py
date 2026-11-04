#!/usr/bin/env python3
"""交互式录入一条 V2 场景对话，追加到 data/v2/scenarios/<scenario>.jsonl。

用法:
  python scripts/07_v2_entry.py
  python scripts/07_v2_entry.py --scenario daily
  python scripts/07_v2_entry.py --scenario comfort --turns 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
SYSTEM = (V2 / "SYSTEM_TEMPLATE.txt").read_text(encoding="utf-8").strip()
SCENARIOS = [
    "daily",
    "comfort",
    "robot_boundary",
    "safety_refuse",
    "safety_allow",
    "multi_turn",
    "lore_identity",
    "with_memory",
]


def prompt(msg: str, default: str = "") -> str:
    suf = f" [{default}]" if default else ""
    s = input(f"{msg}{suf}: ").strip()
    return s or default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS)
    ap.add_argument("--turns", type=int, default=0, help="human/gpt 轮数，0=交互询问")
    ap.add_argument("--memory", default="", help="with_memory 时写入 system 的记忆块")
    args = ap.parse_args()

    print("ATRI V2 录入工具 — 空行结束多行输入时用单行即可")
    scen = args.scenario or prompt("场景 " + "/".join(SCENARIOS), "daily")
    if scen not in SCENARIOS:
        print("未知场景", file=sys.stderr)
        return 1

    n = args.turns or int(prompt("对话轮数（一对 human+gpt 算 1 轮）", "1"))
    n = max(1, min(n, 12))
    conv = []
    for i in range(n):
        h = prompt(f"用户[{i+1}]")
        g = prompt(f"ATRI[{i+1}]")
        if not h or not g:
            print("不能为空", file=sys.stderr)
            return 1
        conv.append({"from": "human", "value": h.replace("\\n", "\n")})
        conv.append({"from": "gpt", "value": g.replace("\\n", "\n")})

    tags = [t for t in prompt("标签（逗号分隔）", scen).split(",") if t.strip()]
    mem = args.memory or (prompt("记忆块（可空）", "") if scen == "with_memory" else "")
    system = SYSTEM
    if mem.strip():
        # 替换记忆占位为实填（训练时不用 {{memory}}）
        system = system.replace("{{memory}}", mem.strip())

    ctx = "with_memory" if scen == "with_memory" or mem else ("multi_turn" if n >= 3 else "cold_start")
    row = {
        "system": system,
        "conversations": conv,
        "meta": {
            "scenario": scen,
            "tags": tags,
            "ctx": ctx,
            "quality": {"character": True, "response": True, "consistency": True},
            "source": "authored",
            "split": "pilot",
        },
    }
    out = V2 / "scenarios" / f"{scen}.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已追加 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
