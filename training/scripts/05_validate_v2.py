#!/usr/bin/env python3
"""校验 data/v2 场景样本：schema + 质检三维度 + 双人交替。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
SCENARIOS = V2 / "scenarios"

ALLOWED_SCENARIOS = {
    "daily",
    "comfort",
    "robot_boundary",
    "safety_refuse",
    "safety_allow",
    "multi_turn",
    "lore_identity",
    "with_memory",
}


def validate_row(obj: dict, path: str, line_no: int) -> list[str]:
    errs: list[str] = []
    loc = f"{path}:{line_no}"
    if "system" not in obj or not str(obj["system"]).strip():
        errs.append(f"{loc} missing system")
    conv = obj.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2:
        errs.append(f"{loc} conversations need >= 2 turns")
        return errs
    if len(conv) % 2 != 0:
        errs.append(f"{loc} conversations should be even (human/gpt pairs)")
    for i, m in enumerate(conv):
        expect = "human" if i % 2 == 0 else "gpt"
        if m.get("from") != expect:
            errs.append(f"{loc} turn {i} expected from={expect}, got {m.get('from')}")
        if not str(m.get("value") or "").strip():
            errs.append(f"{loc} turn {i} empty value")
        # 禁止多人剧本糊成一条 human
        val = str(m.get("value") or "")
        if m.get("from") == "human" and val.count("[") >= 3 and "]," in val.replace(" ", ""):
            errs.append(f"{loc} turn {i} looks like multi-speaker script dump")
    meta = obj.get("meta") or {}
    scen = meta.get("scenario")
    if scen not in ALLOWED_SCENARIOS:
        errs.append(f"{loc} bad scenario={scen}")
    q = meta.get("quality") or {}
    for k in ("character", "response", "consistency"):
        if q.get(k) is not True:
            errs.append(f"{loc} quality.{k} must be true")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=SCENARIOS, help="file or directory of jsonl")
    args = ap.parse_args()
    files: list[Path] = []
    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.glob("*.jsonl"))
    if not files:
        print("no jsonl found", file=sys.stderr)
        return 1
    total = 0
    bad = 0
    all_errs: list[str] = []
    by_scen: dict[str, int] = {}
    for fp in files:
        for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                all_errs.append(f"{fp.name}:{i} json error {e}")
                continue
            errs = validate_row(obj, fp.name, i)
            if errs:
                bad += 1
                all_errs.extend(errs)
            else:
                scen = obj["meta"]["scenario"]
                by_scen[scen] = by_scen.get(scen, 0) + 1
    print(f"checked={total} ok={total - bad} bad={bad}")
    print("by_scenario:", json.dumps(by_scen, ensure_ascii=False))
    for e in all_errs[:40]:
        print("ERR", e)
    if len(all_errs) > 40:
        print(f"... {len(all_errs) - 40} more")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
