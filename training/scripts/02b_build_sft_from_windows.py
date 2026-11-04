"""把官方公开剧本清洗后的对白窗口打成可训练 ShareGPT（+ system + safety_seed）。
用法:
  python scripts/02b_build_sft_from_windows.py
输入默认: private/derived/atri_dialogue_windows.jsonl（需先跑 01，建议 --drop-late）
输出默认: data/synthetic/atri_synth_sft.jsonl
说明: 剧本公开 ≠ 自动进 git；大 jsonl 仍建议本机训练、按需决定是否提交。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="private/derived/atri_dialogue_windows.jsonl")
    ap.add_argument("--out", default="data/synthetic/atri_synth_sft.jsonl")
    ap.add_argument("--system", default="persona/system_prompt.txt")
    ap.add_argument("--safety", default="data/synthetic/safety_seed.jsonl")
    ap.add_argument("--stages", default="early,mid", help="comma stages to keep; empty=all")
    a = ap.parse_args()

    system = Path(a.system).read_text(encoding="utf8").strip()
    allow = {s.strip() for s in a.stages.split(",") if s.strip()}
    kept: list[dict] = []
    stats = {"in": 0, "stage_drop": 0, "kept": 0}

    for line in Path(a.inp).read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        stats["in"] += 1
        o = json.loads(line)
        if allow and o.get("stage") not in allow:
            stats["stage_drop"] += 1
            continue
        kept.append(
            {
                "system": system,
                "conversations": o["conversations"],
                "meta": {"stage": o.get("stage"), "heavy": o.get("heavy"), "n_turns": o.get("n_turns")},
                "source": "public_script",
            }
        )
        stats["kept"] += 1

    if Path(a.safety).exists():
        for line in Path(a.safety).read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            kept.append(
                {
                    "system": system,
                    "conversations": o["conversations"],
                    "meta": {"scene": "safety"},
                    "source": "handwritten",
                }
            )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8") as f:
        for o in kept:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print({**stats, "safety_merged": True, "total": len(kept), "out": str(out)})


if __name__ == "__main__":
    main()
