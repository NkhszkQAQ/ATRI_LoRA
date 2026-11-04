"""24: 按「情境模板」重建训练数据（v7）。

主人校准后的角色模型：
- 她的反应是「固定设定反应」（被骂废物→反歧视法/蔑称那套），语境改变的是**语气**不是内容
- 语气按关系亲疏：熟人=轻松/耍宝；陌生人/正式场合=正常/郑重
- 早期不熟=正常客气；后期熟悉=轻松。语气基准=刷牙段（b111 sc5）

数据模板（每条样本）：
- system = 现行 persona（称呼填实）+ [当前场景]（完整旁白，不截断）
- conversations = 2~4 轮「对她说的话 ↔ 她的反应」
- 只保留两类说话人当 user：夏生（熟人）+ 反派/陌生人（安田、废品店老板）
- 其他熟人（水菜萌、龙司、凯瑟琳……）的话转成场景背景
- meta 标注：剧情阶段、触发词、说话人身份（熟人/陌生人）——只进 meta，不进 system

用法：
  python training/scripts/24_build_situational_data.py --sample 3     # 看样例
  python training/scripts/24_build_situational_data.py --write        # 落盘
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

ASSISTANT_ROLES = {"亚托莉", "机器人少女", "少女"}
USER_FRIEND = "夏生"
USER_STRANGERS = {"安田", "废品店老板"}   # 反派/陌生人当 user（练「陌生人→正常语气」）
ALIASES = {"竜司": "龙司", "凜凜花": "凛凛花", "凜凛花": "凛凛花"}
NAMES = ["小林", "小周", "阿霖", "小许", "阿远", "小唐", "小苏", "阿凯", "小程", "阿哲", "阿辉", "阿岚"]

STAGE_MAP = {"1": "早期_初遇", "2": "中期_磨合", "3": "后期_深入", "4": "后期_危机",
             "5": "结局_幸福", "6": "结局_坏", "7": "结局_真", "9": "序章"}

TRIGGERS = {
    "废物": "被骂废物", "蔑称": "被叫蔑称", "歧视": "反歧视法", "撤回": "要求撤回",
    "高性能": "高性能", "螃蟹": "螃蟹", "职责": "主人职责", "战斗": "战斗机器人",
    "三原则": "三原则", "机器人": "机器人身份",
}

EFFECTIVE = re.compile(r"[\w\u4e00-\u9fff]")
SPACE = re.compile(r"[ \t]{2,}")


def stage_of(fname: str) -> str:
    return STAGE_MAP.get(fname[1], "其他")


def clean_text(t: str) -> str:
    t = t.replace("\u3000", " ").strip()
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"[（(][^（）)]{1,24}[）)]", "", t)   # 去括号动作/心理描述
    return SPACE.sub(" ", t)


def is_noise(t: str) -> bool:
    chars = EFFECTIVE.findall(t)
    if len(chars) < 2:
        return True
    if len(chars) >= 6 and len(set(chars)) / len(chars) < 0.4:
        return True
    return False


def replace_name(t: str, name: str) -> str:
    return t.replace(f"{USER_FRIEND}先生", f"{name}先生").replace(USER_FRIEND, name)


class Turn:
    __slots__ = ("role", "text", "speaker", "narration")

    def __init__(self, role, text, speaker, narration):
        self.role = role
        self.text = text
        self.speaker = speaker
        self.narration = narration


def parse_scene(dialogues, name: str) -> tuple[list[Turn], str]:
    """场景 → 交替 turns + 场景背景文本。返回 (turns, background)"""
    turns: list[Turn] = []
    pending: list[str] = []
    bg_parts: list[str] = []

    for d in dialogues:
        raw = (d.get("character") or "").strip()
        text = clean_text(d.get("text") or "")
        if not text:
            continue
        if not raw:                                   # 旁白
            if not is_noise(text):
                pending.append(text)
                bg_parts.append(text)
            continue
        char = ALIASES.get(raw, raw)
        if char in ASSISTANT_ROLES:
            seg = replace_name(text, name)
            if turns and turns[-1].role == "gpt":
                turns[-1].text += "\n" + seg
            else:
                turns.append(Turn("gpt", seg, "亚托莉", pending))
                pending = []
        elif char == USER_FRIEND:
            seg = replace_name(text, name)
            if turns and turns[-1].role == "human":
                turns[-1].text += " " + seg
            else:
                turns.append(Turn("human", seg, "熟人", pending))
                pending = []
        elif char in USER_STRANGERS:
            if turns and turns[-1].role == "human":
                turns[-1].text += " " + text
            else:
                turns.append(Turn("human", text, "陌生人", pending))
                pending = []
        else:                                          # 其他熟人 → 场景背景
            bg_parts.append(f"[{char}] {replace_name(text, name)}")

    background = " ".join(bg_parts)
    return turns, background


def make_record(win_turns, system, scene_bg, meta) -> dict:
    convs = [{"from": "gpt" if t.role == "gpt" else "human", "value": t.text} for t in win_turns]
    sys_text = system
    if scene_bg:
        sys_text = system.rstrip() + f"\n\n[当前场景]\n{scene_bg}\n"
    return {"system": sys_text, "conversations": convs, "meta": meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", default="/tmp/atri_script")
    ap.add_argument("--persona", default="persona/system_prompt.txt")
    ap.add_argument("--out", default="training/data/v2/pilot/atri_situational_v7.jsonl")
    ap.add_argument("--min-rounds", type=int, default=2)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    persona = Path(a.persona).read_text(encoding="utf8").strip()
    files = sorted(glob.glob(str(Path(a.script_dir) / "*.txt")))
    if not files:
        raise SystemExit(f"找不到剧本文本：{a.script_dir}/*.txt")

    rng = random.Random(a.seed)
    records: list[dict] = []
    stats = Counter()

    for path in files:
        fname = Path(path).name
        stage = stage_of(fname)
        data = json.load(open(path, encoding="utf8"))
        for sc in data:
            dialogues = sc.get("dialogues") or []
            if not any((d.get("character") or "").strip() in ASSISTANT_ROLES for d in dialogues):
                continue
            name = rng.choice(NAMES)
            turns, background = parse_scene(dialogues, name)
            # 窗口：每个 assistant 回合向前取 2~4 轮
            for k, t in enumerate(turns):
                if t.role != "gpt" or is_noise(t.text):
                    continue
                rounds = rng.randint(a.min_rounds, a.max_rounds)
                start = max(0, k - (rounds * 2 - 1))
                while start < k and turns[start].role != "human":
                    start += 1
                win = turns[start:k + 1]
                if len(win) < 2 or win[0].role != "human" or is_noise(win[0].text):
                    continue
                # 取窗口起点之前的旁白（最近 2 条，完整）
                nar = []
                for i in range(start, -1, -1):
                    for line in reversed(turns[i].narration):
                        if len(nar) >= 2:
                            break
                        nar.append(line)
                    if len(nar) >= 2:
                        break
                nar = list(reversed(nar))
                scene_text = " ".join(nar) if nar else ""
                triggers = [label for kw, label in TRIGGERS.items() if kw in t.text]
                meta = {
                    "stage": stage,
                    "speaker": win[-2].speaker,
                    "n_rounds": sum(1 for x in win if x.role == "human"),
                    "triggers": triggers,
                    "source_file": fname,
                }
                sys_text = (persona.replace("{{user}}", name)
                                   .replace("{{honorific}}", "先生")
                                   .replace("{{memory}}", "（暂无）"))
                records.append(make_record(win, sys_text, scene_text, meta))
                stats["total"] += 1
                stats[f"stage:{stage}"] += 1
                if triggers:
                    for tg in triggers:
                        stats[f"trigger:{tg}"] += 1

    print("生成样本:", stats["total"])
    print("阶段分布:", {k.split(':')[1]: v for k, v in stats.items() if k.startswith("stage:")})
    print("触发词分布:", {k.split(':')[1]: v for k, v in sorted(stats.items()) if k.startswith("trigger:")})

    if a.sample:
        # 挑含触发词的样例 + 一条陌生人的
        picks = [r for r in records if r["meta"]["triggers"]] or records
        for r in rng.sample(picks, min(a.sample, len(picks))):
            print("\n" + "=" * 70)
            print("【阶段】", r["meta"]["stage"], "| 说话人:", r["meta"]["speaker"], "| 触发:", r["meta"]["triggers"])
            tail = r["system"][r["system"].rfind("[当前场景]"):]
            print("【system 场景段】", tail[:160].replace("\n", " ") if tail else "（无场景）")
            for c in r["conversations"]:
                print(f"  {c['from']:5s} | {c['value'][:80]}")

    if a.write:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("\n已写出:", out, "|", len(records), "条")


if __name__ == "__main__":
    main()
