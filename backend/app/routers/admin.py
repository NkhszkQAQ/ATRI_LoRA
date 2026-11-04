from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session, joinedload

from app.admin_serialize import audit_out, conversation_out, message_out, safety_event_out, user_label
from app.db import get_db
from app.deps import audit, require_admin
from app.models import AuditLog, Conversation, Message, SafetyEvent, SystemSetting, User, UserRole
from app.ollama_client import list_models
from app.schemas import (
    AdminUserPatch,
    AuditOut,
    ConversationOut,
    MessageOut,
    ModelSettingIn,
    SafetyEventOut,
    SmtpSettingIn,
    UserOut,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/users", response_model=list[UserOut])
def users(
    q: Optional[str] = None,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if q:
        like = f"%{q}%"
        query = query.filter((User.email.ilike(like)) | (User.display_name.ilike(like)))
    return query.order_by(User.id.desc()).limit(200).all()


@router.patch("/users/{uid}", response_model=UserOut)
def patch_user(
    uid: int,
    body: AdminUserPatch,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404, "用户不存在")
    if body.is_active is not None:
        u.is_active = body.is_active
    if body.role is not None:
        u.role = UserRole(body.role)
    if body.display_name is not None:
        u.display_name = body.display_name
    if body.honorific is not None:
        u.honorific = body.honorific
    if body.memory_note is not None:
        from app.memory import set_user_memory

        set_user_memory(db, u, body.memory_note)
    db.commit()
    db.refresh(u)
    name, _ = user_label(u)
    audit(
        db,
        actor=admin,
        action="admin.user.patch",
        target_type="user",
        target_id=name,
        detail={**body.model_dump(exclude={"memory_note"}), "target_name": name, "target_email": u.email},
        request=request,
    )
    return u


@router.get("/users/by-name/{name}/memory")
def get_user_memory_by_name(
    name: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """按显示名或邮箱前缀查找记忆（避免后台暴露用户数字 ID）。"""
    like = f"%{name}%"
    u = (
        db.query(User)
        .filter(or_(User.display_name.ilike(like), User.email.ilike(like)))
        .order_by(User.id.desc())
        .first()
    )
    if not u:
        raise HTTPException(404, "用户不存在")
    uname, email = user_label(u)
    return {"user_name": uname, "user_email": email, "memory_note": u.memory_note or ""}


@router.get("/conversations", response_model=list[ConversationOut])
def conversations(
    user: Optional[str] = None,
    model: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = db.query(Conversation).options(joinedload(Conversation.user))
    filters = []
    if user:
        like = f"%{user}%"
        query = query.join(User, User.id == Conversation.user_id).filter(
            or_(User.display_name.ilike(like), User.email.ilike(like))
        )
    if model:
        filters.append(Conversation.model == model)
    if date_from:
        filters.append(Conversation.created_at >= date_from)
    if date_to:
        filters.append(Conversation.created_at <= date_to)
    if filters:
        query = query.filter(and_(*filters))
    rows = query.order_by(Conversation.updated_at.desc()).limit(300).all()
    return [conversation_out(c) for c in rows]


@router.get("/conversations/{cid}/messages", response_model=list[MessageOut])
def conversation_messages(
    cid: int,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    c = db.get(Conversation, cid)
    if not c:
        raise HTTPException(404, "对话不存在")
    owner = db.get(User, c.user_id)
    name, email = user_label(owner)
    audit(
        db,
        actor=admin,
        action="admin.chat.view",
        target_type="conversation",
        target_id=str(cid),
        detail={"user_name": name, "user_email": email, "model": c.model, "title": c.title},
        request=request,
    )
    rows = db.query(Message).filter(Message.conversation_id == cid).order_by(Message.id.asc()).all()
    return [message_out(m) for m in rows]


@router.get("/safety-events", response_model=list[SafetyEventOut])
def safety_events(
    category: Optional[str] = None,
    severity: Optional[str] = None,
    user: Optional[str] = None,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = db.query(SafetyEvent)
    if category:
        query = query.filter(SafetyEvent.category == category)
    if severity:
        query = query.filter(SafetyEvent.severity == severity)
    if user:
        like = f"%{user}%"
        query = query.join(User, User.id == SafetyEvent.user_id).filter(
            or_(User.display_name.ilike(like), User.email.ilike(like))
        )
    rows = query.order_by(SafetyEvent.id.desc()).limit(300).all()
    return [safety_event_out(db, ev) for ev in rows]


@router.get("/safety-events/{eid}", response_model=SafetyEventOut)
def safety_event_detail(
    eid: int,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    ev = db.get(SafetyEvent, eid)
    if not ev:
        raise HTTPException(404, "事件不存在")
    out = safety_event_out(db, ev)
    audit(
        db,
        actor=admin,
        action="admin.safety.view_raw",
        target_type="safety_event",
        target_id=str(eid),
        detail={"category": out.category, "user_name": out.user_name, "user_email": out.user_email},
        request=request,
    )
    return out


@router.get("/audit-logs", response_model=list[AuditOut])
def audit_logs(
    action: Optional[str] = None,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = db.query(AuditLog)
    if action:
        query = query.filter(AuditLog.action == action)
    rows = query.order_by(AuditLog.id.desc()).limit(300).all()
    return [audit_out(db, row) for row in rows]


@router.get("/models")
async def models(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    from app.config import get_settings

    local = []
    try:
        local = await list_models()
    except Exception as e:
        local = [f"(ollama unreachable: {e})"]
    row = db.query(SystemSetting).filter(SystemSetting.key == "model").first()
    active = (row.value.get("model") if row and row.value else None) or get_settings().atri_model
    return {"active": active, "available": local}


@router.put("/models")
def set_model(
    body: ModelSettingIn,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    row = db.query(SystemSetting).filter(SystemSetting.key == "model").first()
    if not row:
        row = SystemSetting(key="model", value={"model": body.model})
        db.add(row)
    else:
        row.value = {"model": body.model}
    db.commit()
    audit(db, actor=admin, action="admin.model.set", target_type="model", target_id=body.model, request=request)
    return {"ok": True, "model": body.model}


@router.get("/smtp")
def get_smtp(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    from app.config import get_settings

    s = get_settings()
    row = db.query(SystemSetting).filter(SystemSetting.key == "smtp").first()
    base = {
        "enabled": s.smtp_enabled,
        "host": s.smtp_host,
        "port": s.smtp_port,
        "user": s.smtp_user,
        "password": "***" if s.smtp_password else "",
        "from_addr": s.smtp_from,
        "tls": s.smtp_tls,
    }
    if row and row.value:
        merged = {**base, **row.value}
        if merged.get("password"):
            merged["password"] = "***"
        return merged
    return base


@router.put("/smtp")
def set_smtp(
    body: SmtpSettingIn,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    data = body.model_dump()
    row = db.query(SystemSetting).filter(SystemSetting.key == "smtp").first()
    if not row:
        db.add(SystemSetting(key="smtp", value=data))
    else:
        if data.get("password") in ("", "***") and row.value and row.value.get("password"):
            data["password"] = row.value["password"]
        row.value = data
    db.commit()
    audit(
        db,
        actor=admin,
        action="admin.smtp.set",
        target_type="smtp",
        target_id="smtp",
        detail={"enabled": body.enabled},
        request=request,
    )
    return {"ok": True}
