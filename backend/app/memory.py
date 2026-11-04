"""跨会话记忆：写入 / 合并 / 注入。

记忆不靠 LoRA「记住」，由本模块维护 User.memory_note，
经 persona_runtime 注入 system 的 {{memory}} 块。
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import User

MAX_MEMORY_CHARS = 1200
MAX_BULLETS = 12

# 从用户话里抽取可长期保留的事实（启发式；后续可换 LLM 摘要）
_FACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:我叫|我的名字是|叫我)\s*([^\s，。！？,]{1,20})"), "称呼相关：用户提到名字 {0}"),
    (re.compile(r"(?:喜欢|爱上了|超爱)\s*([^\s，。！？]{1,30})"), "喜好：{0}"),
    (re.compile(r"(?:讨厌|不喜欢|超烦)\s*([^\s，。！？]{1,30})"), "厌恶：{0}"),
    (re.compile(r"(?:昨天|昨晚).{0,8}(没睡|失眠|睡得很晚|熬夜)"), "近期：睡眠不足"),
    (re.compile(r"(?:今天|这会儿).{0,6}(好累|很累|加班|压力大)"), "近期：疲劳/压力"),
    (re.compile(r"(?:住在|在)\s*([\u4e00-\u9fff]{2,12})(?:市|城|镇)"), "地点：{0}"),
]


def format_memory_block(note: str | None) -> str:
    text = (note or "").strip()
    return text if text else "（暂无）"


def parse_bullets(note: str | None) -> list[str]:
    lines = []
    for raw in (note or "").splitlines():
        s = raw.strip().lstrip("-•* ").strip()
        if s and s != "（暂无）":
            lines.append(s)
    return lines


def render_bullets(bullets: list[str]) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for b in bullets:
        b = re.sub(r"\s+", " ", b).strip()
        if not b or b in seen:
            continue
        seen.add(b)
        cleaned.append(b)
        if len(cleaned) >= MAX_BULLETS:
            break
    text = "\n".join(f"- {b}" for b in cleaned)
    if len(text) > MAX_MEMORY_CHARS:
        text = text[: MAX_MEMORY_CHARS - 1].rstrip() + "…"
    return text


def extract_fact_candidates(user_text: str) -> list[str]:
    facts: list[str] = []
    t = (user_text or "").strip()
    if not t:
        return facts
    for pat, tmpl in _FACT_PATTERNS:
        m = pat.search(t)
        if m:
            try:
                facts.append(tmpl.format(*m.groups()) if m.groups() else tmpl)
            except Exception:
                facts.append(tmpl)
    return facts


def merge_memory(existing: str | None, new_facts: list[str]) -> str:
    bullets = parse_bullets(existing)
    # 新事实插到前面（更新鲜）
    for f in reversed(new_facts):
        f = f.strip()
        if not f:
            continue
        # 同前缀去重（如「喜好：」类）
        prefix = f.split("：")[0] + "：" if "：" in f else ""
        if prefix:
            bullets = [b for b in bullets if not b.startswith(prefix)]
        if f not in bullets:
            bullets.insert(0, f)
    if new_facts:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stamp_line = f"上次更新：{stamp}"
        bullets = [b for b in bullets if not b.startswith("上次更新：")]
        bullets.append(stamp_line)
    return render_bullets(bullets)


def update_user_memory(db: Session, user: User, user_text: str) -> bool:
    """根据本轮用户发言更新 memory_note。有变更返回 True。"""
    facts = extract_fact_candidates(user_text)
    if not facts:
        return False
    merged = merge_memory(user.memory_note, facts)
    if merged == (user.memory_note or "").strip():
        return False
    user.memory_note = merged
    db.add(user)
    db.commit()
    return True


def set_user_memory(db: Session, user: User, note: str) -> str:
    """管理员/用户手动覆写记忆（规范化为条目列表）。"""
    bullets = parse_bullets(note)
    # 若用户传入纯段落，整段保留为一条
    if not bullets and note.strip():
        bullets = [note.strip()[:400]]
    text = render_bullets(bullets)
    user.memory_note = text
    db.add(user)
    db.commit()
    db.refresh(user)
    return user.memory_note or ""
