from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import get_current_user
from app.memory import set_user_memory, update_user_memory
from app.models import Conversation, Message, SafetyEvent, User
from app.ollama_client import chat_stream
from app.persona_runtime import load_system_prompt
from app.request_meta import build_token_info, client_ip, client_ua
from app.safety import CRISIS_REPLY, PROMPT_BLOCK_REPLY, scan_user_text
from app.schemas import ChatSendIn, ConversationOut, MemoryIn, MemoryOut, MessageOut

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get("/memory", response_model=MemoryOut)
def get_memory(user: User = Depends(get_current_user)):
    return MemoryOut(memory_note=user.memory_note or "")


@router.put("/memory", response_model=MemoryOut)
def put_memory(
    body: MemoryIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    note = set_user_memory(db, user, body.memory_note)
    return MemoryOut(memory_note=note)


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(100)
        .all()
    )
    return rows


@router.post("/conversations", response_model=ConversationOut)
def create_conversation(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    settings = get_settings()
    model = _active_model(db, settings.atri_model)
    c = Conversation(user_id=user.id, title="新对话", model=model)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@router.get("/conversations/{cid}/messages", response_model=list[MessageOut])
def get_messages(cid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    c = db.get(Conversation, cid)
    if not c or c.user_id != user.id:
        raise HTTPException(404, "对话不存在")
    return (
        db.query(Message)
        .filter(Message.conversation_id == cid)
        .order_by(Message.id.asc())
        .all()
    )


def _active_model(db: Session, fallback: str) -> str:
    from app.models import SystemSetting

    row = db.query(SystemSetting).filter(SystemSetting.key == "model").first()
    if row and isinstance(row.value, dict) and row.value.get("model"):
        return str(row.value["model"])
    return fallback


@router.post("/send")
async def send_message(
    body: ChatSendIn,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    model = _active_model(db, settings.atri_model)

    if body.conversation_id:
        conv = db.get(Conversation, body.conversation_id)
        if not conv or conv.user_id != user.id:
            raise HTTPException(404, "对话不存在")
    else:
        conv = Conversation(user_id=user.id, title=body.content[:24] or "新对话", model=model)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    hit = scan_user_text(body.content)
    user_msg = Message(conversation_id=conv.id, role="user", content=body.content, model=model)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    # 跨会话记忆：从用户话抽取长期事实并写入 memory_note（注入下一轮 system）
    if not hit or hit.action == "logged":
        try:
            update_user_memory(db, user, body.content)
            db.refresh(user)
        except Exception:
            pass

    if hit:
        ip = client_ip(request)
        ua = client_ua(request)
        token_info = build_token_info(
            request,
            model=model,
            enable_thinking=bool(body.enable_thinking),
            extra={
                "safety_action": hit.action,
                "safety_category": hit.category.value,
                "matched": hit.matched,
            },
        )

        if hit.action == "blocked":
            reply = PROMPT_BLOCK_REPLY
        elif hit.action == "redirected":
            reply = CRISIS_REPLY.replace("{{user}}", user.display_name or "用户")
        else:
            reply = ""

        ev = SafetyEvent(
            user_id=user.id,
            conversation_id=conv.id,
            message_id=user_msg.id,
            category=hit.category,
            severity=hit.severity,
            probability=hit.probability,
            language_hint=hit.language_hint,
            raw_text=body.content,
            action=hit.action,
            ip=ip,
            user_agent=ua,
            model=model,
            reply_text=reply,
            reasoning_text="",
            token_info=token_info,
        )
        db.add(ev)
        db.commit()

        if hit.action == "blocked":
            asst = Message(
                conversation_id=conv.id,
                role="assistant",
                content=reply,
                model=model,
                meta={"safety": hit.category.value},
            )
            db.add(asst)
            db.commit()

            async def blocked_stream() -> AsyncIterator[str]:
                payload = {
                    "conversation_id": conv.id,
                    "content": reply,
                    "reasoning": "",
                    "done": True,
                    "blocked": True,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            return StreamingResponse(blocked_stream(), media_type="text/event-stream")

        if hit.action == "redirected":
            asst = Message(
                conversation_id=conv.id,
                role="assistant",
                content=reply,
                model=model,
                meta={"safety": hit.category.value},
            )
            db.add(asst)
            db.commit()

            async def crisis_stream() -> AsyncIterator[str]:
                payload = {
                    "conversation_id": conv.id,
                    "content": reply,
                    "reasoning": "",
                    "done": True,
                    "crisis": True,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            return StreamingResponse(crisis_stream(), media_type="text/event-stream")

    history = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.id.asc())
        .all()
    )
    system = load_system_prompt(user)
    ollama_msgs = [{"role": "system", "content": system}]
    for m in history:
        if m.role in ("user", "assistant"):
            ollama_msgs.append({"role": m.role, "content": m.content})

    if not conv.title or conv.title == "新对话":
        conv.title = body.content[:24]
    conv.model = model
    db.commit()

    conv_id = conv.id
    enable_thinking = bool(body.enable_thinking)

    async def gen() -> AsyncIterator[str]:
        from app.db import SessionLocal

        content_buf: list[str] = []
        reason_buf: list[str] = []
        try:
            async for chunk in chat_stream(
                model=model, messages=ollama_msgs, enable_thinking=enable_thinking
            ):
                c = chunk.get("content") or ""
                r = chunk.get("reasoning") or ""
                if c:
                    content_buf.append(c)
                if r:
                    reason_buf.append(r)
                yield f"data: {json.dumps({'conversation_id': conv_id, 'content': c, 'reasoning': r, 'done': False}, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = f"（连接模型失败：{e}）"
            content_buf.append(err)
            yield f"data: {json.dumps({'conversation_id': conv_id, 'content': err, 'reasoning': '', 'done': False}, ensure_ascii=False)}\n\n"
        full = "".join(content_buf)
        reasoning = "".join(reason_buf)
        s = SessionLocal()
        try:
            meta = {"reasoning": reasoning} if reasoning else None
            s.add(
                Message(
                    conversation_id=conv_id,
                    role="assistant",
                    content=full,
                    model=model,
                    meta=meta,
                )
            )
            s.commit()
        finally:
            s.close()
        yield f"data: {json.dumps({'conversation_id': conv_id, 'content': '', 'reasoning': '', 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
