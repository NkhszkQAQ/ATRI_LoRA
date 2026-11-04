from __future__ import annotations

import random
import string
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import audit, get_current_user
from app.mailer import send_email
from app.models import EmailVerification, User, UserRole
from app.schemas import LoginIn, RegisterIn, SendCodeIn, TokenOut, UserOut
from app.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _code(n: int = 6) -> str:
    return "".join(random.choices(string.digits, k=n))


@router.post("/send-code")
def send_code(body: SendCodeIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "邮箱已注册")
    code = _code()
    db.add(
        EmailVerification(
            email=body.email,
            code=code,
            purpose=body.purpose,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )
    db.commit()
    send_email(
        body.email,
        "ATRI LoRA 邮箱验证码",
        f"你的验证码是：{code}，15 分钟内有效。\n如非本人操作请忽略。",
        db=db,
    )
    return {"ok": True, "message": "验证码已发送（若未启用 SMTP 请查看 API 日志）"}


@router.post("/register", response_model=TokenOut)
def register(body: RegisterIn, request: Request, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "邮箱已注册")
    ver = (
        db.query(EmailVerification)
        .filter(
            EmailVerification.email == body.email,
            EmailVerification.purpose == "register",
            EmailVerification.used.is_(False),
        )
        .order_by(EmailVerification.id.desc())
        .first()
    )
    if not ver or ver.code != body.code:
        raise HTTPException(400, "验证码错误")
    if ver.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(400, "验证码已过期")
    ver.used = True
    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        display_name=body.display_name or body.email.split("@")[0],
        role=UserRole.user,
        email_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    audit(db, actor=user, action="user.register", target_type="user", target_id=str(user.id), request=request)
    return TokenOut(access_token=create_access_token(user.id, {"role": user.role.value}))


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(400, "邮箱或密码错误")
    if not user.is_active:
        raise HTTPException(403, "账号已停用")
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    audit(db, actor=user, action="user.login", target_type="user", target_id=str(user.id), request=request)
    return TokenOut(access_token=create_access_token(user.id, {"role": user.role.value}))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user
