"""22: 从剧本重建「带上下文」的多轮窗口数据（v6）。

背景：v4_clean 99% 是单轮，模型学不到「看前文选语气」，退化成关键词抽台词。
本脚本从 /tmp/atri_script 的 34 个剧本文本重建 2~4 轮窗口，并把旁白变成场景信息。

用法（先看样例，不落盘）：
    python training/scripts/22_build_context_windows.py --sample 3
批量落盘：
    python training/scripts/22_build_context_windows.py --write

输入：--script-dir 默认 /tmp/atri_script（34 个 .txt，内容是 JSON 数组）
输出：--out 默认 training/data/v2/pilot/atri_windows_v6.jsonl
      每条 = {system, conversations[2~4轮], meta{...}}

旁白两方案（--narration）：
    scene   旁白拼进 system 末尾的「[当前场景]」，推理时也带在 system 里（推荐默认）
    context 旁白作为窗口首条 user 的前置叙述「（……）」，更贴近「前文含旁白」
    both    同时输出两份，便于对比

设计约束（来自 2026-11-01 日志）：
- 只对 assistant（亚托莉）算 loss；user 里可带 [角色名] 前缀区分说话人
- 最后一条 assistant 必须有实际内容（纯符号/省略号不计入样本）
- 称呼：夏生 → 12 个人名池随机；保留原作「X先生」形式
- 不覆盖 v4_clean，输出新文件做 A/B
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

# 角色归一：亚托莉的早期称呼（机器人少女 / 少女）也算她
ASSISTANT_ROLES = {"亚托莉", "机器人少女", "少女"}
USER_ROLE = "夏生"
ALIASES = {"竜司": "龙司", "凜凜花": "凛凛花", "凜凛花": "凛凛花"}

# 与 v4_clean 一致的 12 人名池
NAMES = ["小林", "小周", "阿霖", "小许", "阿远", "小唐", "小苏", "阿凯", "小程", "阿哲", "阿辉", "阿岚"]

EFFECTIVE = re.compile(r"[\w\u4e00-\u9fff]")
SPACE = re.compile(r"[ \t]{2,}")


def clean_text(t: str) -> str:
    t = t.replace("\u3000", " ").strip()
    t = re.sub(r"\s*\n\s*", " ", t)
    return SPACE.sub(" ", t)


def effective_len(t: str) -> int:
    return len(EFFECTIVE.findall(t))


def is_noise(t: str) -> bool:
    """过滤纯符号、纯拟声（咕嘟咕嘟/————）、复读式短串。保留「好的」「小周……」这类真实短句。"""
    chars = EFFECTIVE.findall(t)
    if len(chars) < 2:
        return True
    if len(chars) >= 6 and len(set(chars)) / len(chars) < 0.4:
        return True
    return False


def replace_name(t: str, name: str) -> str:
    """把原作主角名换成随机称呼；先替换「夏生先生」避免残留。"""
    return t.replace(f"{USER_ROLE}先生", f"{name}先生").replace(USER_ROLE, name)


class Turn:
    __slots__ = ("role", "text", "speakers", "narration")

    def __init__(self, role: str, text: str, speakers: list[str], narration: list[str]):
        self.role = role
        self.text = text
        self.speakers = speakers
        self.narration = narration


def parse_scene(dialogues: list[dict], name: str) -> list[Turn]:
    """场景内对白 → 交替的 turns；旁白挂到「它后面第一个 turn」上。"""
    turns: list[Turn] = []
    pending_narration: list[str] = []
    for d in dialogues:
        raw_char = (d.get("character") or "").strip()
        text = clean_text(d.get("text") or "")
        if not text:
            continue
        if not raw_char:                      # 旁白
            pending_narration.append(text)
            continue
        char = ALIASES.get(raw_char, raw_char)
        if char in ASSISTANT_ROLES:
            role = "gpt"
            seg = replace_name(text, name)
        else:
            role = "human"
            prefix = "" if char == USER_ROLE else f"[{char}] "
            seg = prefix + replace_name(text, name)
        if turns and turns[-1].role == role and role == "gpt":
            turns[-1].text += "\n" + seg
            turns[-1].speakers.append(char)
        elif turns and turns[-1].role == role and role == "human":
            turns[-1].text += " " + seg
            turns[-1].speakers.append(char)
        else:
            turns.append(Turn(role, seg, [char], pending_narration))
            pending_narration = []
    return turns


def pick_narration(turns: list[Turn], start: int, max_chars: int = 60) -> str:
    """取窗口起点之前最近的有内容旁白（1 句）。"""
    for i in range(start, -1, -1):
        for line in reversed(turns[i].narration):
            if effective_len(line) >= 3:
                return line if len(line) <= max_chars else line[:max_chars] + "……"
    return ""


def build_windows(turns: list[Turn], name: str, min_rounds: int, max_rounds: int,
                  rng: random.Random):
    """每个 assistant turn 生成一条窗口：向前取 2~4 轮（1 轮 = 1 问 1 答）。"""
    out = []
    for k, t in enumerate(turns):
        if t.role != "gpt":
            continue
        if is_noise(t.text):                   # 纯「……」/拟声/复读的当前轮不要
            continue
        rounds = rng.randint(min_rounds, max_rounds)
        start = max(0, k - (rounds * 2 - 1))
        while start < k and turns[start].role != "human":
            start += 1
        window = turns[start:k + 1]
        if len(window) < 2 or window[0].role != "human" or window[-1].role != "gpt":
            continue
        if is_noise(window[0].text):
            continue
        out.append({
            "turns": window,
            "start": start,
            "k": k,
            "name": name,
            "narration": pick_narration(turns, start),
        })
    return out


def make_record(win: dict, system: str, narration_mode: str) -> dict:
    turns = win["turns"]
    convs = [{"from": "gpt" if x.role == "gpt" else "human", "value": x.text} for x in turns]
    scene = win.get("narration") or ""
    # 填入占位符
    sys_text = (system.replace("{{user}}", win["name"])
                      .replace("{{honorific}}", "先生")
                      .replace("{{memory}}", "（暂无）"))
    if narration_mode == "scene" and scene:
        sys_text = sys_text.rstrip() + f"\n\n[当前场景]\n{scene}\n"
    elif narration_mode == "context" and scene:
        convs[0] = {"from": "human", "value": f"（{scene}）{convs[0]['value']}"}
    return {
        "system": sys_text,
        "conversations": convs,
        "meta": {
            "source": "public_script_v6",
            "narration_mode": narration_mode,
            "narration": scene,
            "n_turns": len(convs),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", default="/tmp/atri_script")
    ap.add_argument("--persona", default="persona/system_prompt.txt")
    ap.add_argument("--out", default="training/data/v2/pilot/atri_windows_v6.jsonl")
    ap.add_argument("--narration", choices=["scene", "context", "both"], default="scene")
    ap.add_argument("--min-rounds", type=int, default=2, help="每条窗口最少轮数（1 轮=1 问 1 答）")
    ap.add_argument("--max-rounds", type=int, default=4, help="每条窗口最多轮数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", type=int, default=0, help="只打印样例，不落盘")
    ap.add_argument("--write", action="store_true", help="批量写 jsonl")
    a = ap.parse_args()

    persona = Path(a.persona).read_text(encoding="utf8").strip()
    files = sorted(glob.glob(str(Path(a.script_dir) / "*.txt")))
    if not files:
        raise SystemExit(f"找不到剧本文本：{a.script_dir}/*.txt")

    rng = random.Random(a.seed)
    modes = ["scene", "context"] if a.narration == "both" else [a.narration]
    records: list[dict] = []
    stats = Counter()
    scenes_total = 0

    for path in files:
        data = json.load(open(path, encoding="utf8"))
        for si, scene in enumerate(data):
            scenes_total += 1
            dialogues = scene.get("dialogues") or []
            # 只为含 assistant 的场景构造
            if not any((d.get("character") or "").strip() in ASSISTANT_ROLES for d in dialogues):
                stats["scene_no_atri"] += 1
                continue
            name = rng.choice(NAMES)
            turns = parse_scene(dialogues, name)
            for win in build_windows(turns, name, a.min_rounds, a.max_rounds, rng):
                for mode in modes:
                    records.append(make_record(win, persona, mode))
                stats["windows"] += 1

    # 统计
    for r in records:
        n_rounds = sum(1 for c in r["conversations"] if c["from"] == "human")
        stats[f"rounds_{n_rounds}"] += 1
        stats["with_narration"] += 1 if r["meta"]["narration"] else 0
        stats["calls_name"] += 1 if ("主人" in r["conversations"][-1]["value"]
                                     or "先生" in r["conversations"][-1]["value"]) else 0
    n = len(records) or 1
    print("场景总数:", scenes_total, "| 生成样本:", len(records))
    print("轮数分布:", {k: v for k, v in stats.items() if k.startswith("rounds_")})
    print("带旁白比例: %.1f%%" % (100 * stats["with_narration"] / n))
    print("末轮带称呼: %.1f%%" % (100 * stats["calls_name"] / n))

    if a.sample:
        # 优先挑一条「被骂废物」的，直接看语境差异
        picks = [r for r in records if "废物" in r["conversations"][-1]["value"]] or records
        rng2 = random.Random(a.seed)
        for r in rng2.sample(picks, min(a.sample, len(picks))):
            print("\n" + "=" * 72)
            print("【system 末尾】", r["system"].strip().splitlines()[-1] if r["system"].strip() else "")
            print("【narration】", r["meta"]["narration"])
            for c in r["conversations"]:
                print(f"  {c['from']}: {c['value']}")

    if a.write:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("\n已写出:", out, "|", len(records), "条")


if __name__ == "__main__":
    main()
