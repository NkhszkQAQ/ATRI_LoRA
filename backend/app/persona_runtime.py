from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.memory import format_memory_block
from app.models import User


def load_system_prompt(user: User | None = None) -> str:
    settings = get_settings()
    path = Path(settings.system_prompt_path)
    text = path.read_text(encoding="utf-8") if path.exists() else "你是 ATRI。"
    name = (user.display_name if user and user.display_name else "").strip() or "用户"
    # 避免邮箱当显示名时整串过长；仍保持用户设定的 display_name 优先
    if "@" in name and (not user or not (user.display_name or "").strip()):
        name = name.split("@")[0]
    honorific = (user.honorific if user and user.honorific else "先生").strip() or "先生"
    memory = format_memory_block(user.memory_note if user else None)
    # 额外硬锁定，减少 SFT 残留「夏生/斑鸠/Attry」串名
    lock = (
        f"\n\n[本轮硬性约束]\n"
        f"- 你的名字只有 ATRI / 亚托莉。\n"
        f"- 称呼对方只能用「{name}{honorific}」。\n"
        f"- 禁止使用夏生、斑鸠、Attry，或乱编昵称。\n"
    )
    return (
        text.replace("{{user}}", name)
        .replace("{{honorific}}", honorific)
        .replace("{{memory}}", memory)
        + lock
    )
