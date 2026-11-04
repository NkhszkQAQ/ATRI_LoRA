#!/usr/bin/env python3
"""打包 Dataset V2 → pilot / train jsonl，并更新 data/dataset_info.json 条目。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
SCENARIOS = V2 / "scenarios"
MIX = json.loads((V2 / "mix_plan.json").read_text(encoding="utf-8"))


def load_scenario_rows() -> list[dict]:
    rows: list[dict] = []
    for fp in sorted(SCENARIOS.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_legacy(max_n: int) -> list[dict]:
    legacy_dir = V2 / "cleaned_legacy"
    if not legacy_dir.exists():
        return []
    rows: list[dict] = []
    for fp in sorted(legacy_dir.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    random.shuffle(rows)
    return rows[:max_n]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            # LLaMA-Factory 只需 system + conversations；meta 可保留供审计
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_dataset_info(keys: dict[str, str]) -> None:
    info_path = ROOT / "data" / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    for name, rel in keys.items():
        info[name] = {
            "file_name": rel,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
        }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="打包装试训集（默认取全部现有样本）")
    ap.add_argument("--full", action="store_true", help="打包正式训练集")
    ap.add_argument("--mixed", action="store_true", help="正式集 + ≤20% cleaned_legacy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    if not (args.pilot or args.full or args.mixed):
        args.pilot = True

    rows = load_scenario_rows()
    if not rows:
        raise SystemExit("scenarios/ 为空，请先写数据")

    # 按场景统计
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["meta"]["scenario"], []).append(r)

    print("available:", {k: len(v) for k, v in sorted(by.items())})

    if args.pilot:
        pilot_rows = list(rows)
        random.shuffle(pilot_rows)
        out = V2 / "pilot" / "atri_v2_pilot.jsonl"
        write_jsonl(out, pilot_rows)
        ensure_dataset_info({"atri_v2_pilot": "v2/pilot/atri_v2_pilot.jsonl"})
        print(f"wrote {out} n={len(pilot_rows)}")

    if args.full or args.mixed:
        train_rows = list(rows)
        random.shuffle(train_rows)
        out = V2 / "train" / "atri_v2_sft.jsonl"
        write_jsonl(out, train_rows)
        ensure_dataset_info({"atri_v2_sft": "v2/train/atri_v2_sft.jsonl"})
        print(f"wrote {out} n={len(train_rows)}")

    if args.mixed:
        cap = max(1, int(len(rows) * float(MIX["legacy"]["max_mix_ratio"])))
        legacy = load_legacy(cap)
        mixed = list(rows) + legacy
        random.shuffle(mixed)
        out = V2 / "train" / "atri_v2_sft_mixed.jsonl"
        write_jsonl(out, mixed)
        ensure_dataset_info({"atri_v2_sft_mixed": "v2/train/atri_v2_sft_mixed.jsonl"})
        print(f"wrote {out} n={len(mixed)} (legacy={len(legacy)})")

    # 目标缺口提示
    plan = {b["scenario"]: b["count"] for b in MIX["buckets"]}
    # safety_refuse + safety_allow 在 plan 里分开
    print("--- gap vs mix_plan ---")
    counts = {k: len(v) for k, v in by.items()}
    for scen, need in plan.items():
        have = counts.get(scen, 0)
        print(f"  {scen}: {have}/{need}  gap={max(0, need - have)}")


if __name__ == "__main__":
    main()
