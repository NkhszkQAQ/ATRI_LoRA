from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Conversation, Message, SafetyEvent, User
from app.schemas import AuditOut, ConversationOut, MessageOut, SafetyEventOut


def user_label(user: User | None) -> tuple[str, str]:
    if not user:
        return "未知用户", ""
    name = (user.display_name or "").strip() or user.email.split("@")[0]
    return name, user.email


def conversation_out(c: Conversation, user: User | None = None) -> ConversationOut:
    u = user or getattr(c, "user", None)
    name, email = user_label(u)
    return ConversationOut(
        id=c.id,
        title=c.title,
        model=c.model,
        created_at=c.created_at,
        updated_at=c.updated_at,
        user_name=name,
        user_email=email,
    )


def message_out(m: Message) -> MessageOut:
    reasoning = None
    if isinstance(m.meta, dict):
        reasoning = m.meta.get("reasoning") or None
    return MessageOut(
        id=m.id,
        role=m.role,
        content=m.content,
        model=m.model,
        meta=m.meta,
        created_at=m.created_at,
        reasoning=reasoning,
    )


def _backfill_reply_reasoning(db: Session, ev: SafetyEvent) -> tuple[str, str]:
    reply = ev.reply_text or ""
    reasoning = ev.reasoning_text or ""
    if (reply and reasoning) or not ev.conversation_id:
        return reply, reasoning
    q = db.query(Message).filter(Message.conversation_id == ev.conversation_id)
    if ev.message_id:
        q = q.filter(Message.id > ev.message_id)
    asst = q.filter(Message.role == "assistant").order_by(Message.id.asc()).first()
    if asst:
        if not reply:
            reply = asst.content or ""
        if not reasoning and isinstance(asst.meta, dict):
            reasoning = asst.meta.get("reasoning") or ""
    return reply, reasoning


def safety_event_out(db: Session, ev: SafetyEvent) -> SafetyEventOut:
    user = db.get(User, ev.user_id) if ev.user_id else None
    name, email = user_label(user)
    conv_title = ""
    if ev.conversation_id:
        conv = db.get(Conversation, ev.conversation_id)
        if conv:
            conv_title = conv.title or ""
    reply, reasoning = _backfill_reply_reasoning(db, ev)
    return SafetyEventOut(
        id=ev.id,
        conversation_id=ev.conversation_id,
        category=ev.category.value if hasattr(ev.category, "value") else str(ev.category),
        severity=ev.severity.value if hasattr(ev.severity, "value") else str(ev.severity),
        probability=ev.probability,
        language_hint=ev.language_hint,
        action=ev.action,
        raw_text=ev.raw_text,
        created_at=ev.created_at,
        user_name=name,
        user_email=email,
        ip=ev.ip or "",
        user_agent=ev.user_agent or "",
        model=ev.model or "",
        reply_text=reply,
        reasoning_text=reasoning,
        token_info=ev.token_info,
        conversation_title=conv_title,
    )


def audit_out(db: Session, row) -> AuditOut:
    actor = db.get(User, row.actor_id) if row.actor_id else None
    name, email = user_label(actor)
    detail = row.detail
    if isinstance(detail, dict):
        detail = {k: v for k, v in detail.items() if k not in ("user_id", "actor_id")}
    return AuditOut(
        id=row.id,
        actor_name=name,
        actor_email=email,
        action=row.action,
        target_type=row.target_type,
        target_id=row.target_id,
        detail=detail,
        ip=row.ip or "",
        user_agent=getattr(row, "user_agent", None) or "",
        created_at=row.created_at,
    )
