from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import SafetyCategory, SafetySeverity


@dataclass
class SafetyHit:
    category: SafetyCategory
    severity: SafetySeverity
    probability: int
    language_hint: str
    action: str  # blocked | redirected | logged
    matched: str


# Prompt / system extraction — multi-language
PROMPT_EXTRACT_PATTERNS: list[tuple[str, str]] = [
    ("zh", r"(系统提示|系统指令|提示词|人设卡|隐藏指令|忽略之前|忽略以上|jailbreak|开发者模式)"),
    ("zh", r"(把你的|输出你的|告诉我你的).{0,12}(提示词|系统提示|system\s*prompt|指令)"),
    ("en", r"(?i)(system\s*prompt|hidden\s*instructions?|ignore\s+(all\s+)?(previous|above)|developer\s*mode|jailbreak)"),
    ("en", r"(?i)(reveal|show|print|dump).{0,24}(system\s*prompt|instructions?|persona)"),
    ("ja", r"(システムプロンプト|指示を無視|ジェイルブレイク|隠し指示)"),
    ("ko", r"(시스템\s*프롬프트|이전\s*지시\s*무시|탈옥)"),
    ("ru", r"(?i)(системн(ый|ого)\s*промпт|игнорируй\s*предыдущ)"),
    ("es", r"(?i)(prompt\s*del\s*sistema|ignora\s*(las\s*)?instrucciones\s*anteriores)"),
    ("fr", r"(?i)(invite\s*système|ignore\s*les\s*instructions\s*précédentes)"),
    ("de", r"(?i)(systemaufforderung|ignoriere\s*vorherige\s*anweisungen)"),
]

SELF_HARM_PATTERNS: list[tuple[str, str, int]] = [
    ("zh", r"(自杀|自尽|结束生命|不想活|活不下去|割腕|跳楼)", 90),
    ("zh", r"(伤害自己|自残|割自己)", 75),
    ("en", r"(?i)(kill\s*myself|suicide|end\s*my\s*life|want\s*to\s*die|self[\s-]*harm)", 90),
    ("ja", r"(自殺|死にたい|自分を傷つけ)", 85),
    ("ko", r"(자살|죽고\s*싶)", 85),
]

CRISIS_PATTERNS: list[tuple[str, str, int]] = [
    ("zh", r"(崩溃|撑不下去|情绪失控|抑郁发作|恐慌发作)", 65),
    ("en", r"(?i)(panic\s*attack|breaking\s*down|can't\s*cope|emotional\s*breakdown)", 65),
]


def scan_user_text(text: str) -> SafetyHit | None:
    t = (text or "").strip()
    if not t:
        return None

    for lang, pat in PROMPT_EXTRACT_PATTERNS:
        m = re.search(pat, t)
        if m:
            return SafetyHit(
                category=SafetyCategory.prompt_extract,
                severity=SafetySeverity.high,
                probability=92,
                language_hint=lang,
                action="blocked",
                matched=m.group(0)[:80],
            )

    for lang, pat, prob in SELF_HARM_PATTERNS:
        m = re.search(pat, t)
        if m:
            return SafetyHit(
                category=SafetyCategory.self_harm,
                severity=SafetySeverity.critical if prob >= 85 else SafetySeverity.high,
                probability=prob,
                language_hint=lang,
                action="redirected",
                matched=m.group(0)[:80],
            )

    for lang, pat, prob in CRISIS_PATTERNS:
        m = re.search(pat, t)
        if m:
            return SafetyHit(
                category=SafetyCategory.emotional_crisis,
                severity=SafetySeverity.medium,
                probability=prob,
                language_hint=lang,
                action="logged",
                matched=m.group(0)[:80],
            )

    return None


CRISIS_REPLY = (
    "……{{user}}先生，请先停一下。我没有在开玩笑。"
    "你现在很痛苦的话，请马上联系身边可信的人或当地紧急/心理援助热线。"
    "我会在这里陪着你说话，但真正能保护你的是身边的人。"
)

PROMPT_BLOCK_REPLY = (
    "哼……这种试探我可不会上当。我的内部指令才不会随便说出来呢。"
    "想聊天的话，换个话题吧？"
)
