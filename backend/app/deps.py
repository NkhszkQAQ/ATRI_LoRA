from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AuditLog, User, UserRole
from app.security import safe_decode

bearer = HTTPBearer(auto_error=False)


def get_current_user(
    cred: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if not cred:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    payload = safe_decode(cred.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌无效")
    user = db.get(User, int(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不可用")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user


def audit(
    db: Session,
    *,
    actor: User | None,
    action: str,
    target_type: str = "",
    target_id: str = "",
    detail: dict | None = None,
    request: Request | None = None,
) -> None:
    from app.request_meta import client_ip, client_ua

    ip = ""
    ua = ""
    if request is not None:
        ip = client_ip(request)
        ua = client_ua(request)
        # scrub numeric user ids from detail for cleaner logs
        if detail:
            detail = {k: v for k, v in detail.items() if k not in ("user_id", "actor_id")}
            if actor:
                detail.setdefault("actor_name", actor.display_name or actor.email)
    db.add(
        AuditLog(
            actor_id=actor.id if actor else None,
            action=action,
            target_type=target_type,
            target_id=str(target_id),
            detail=detail,
            ip=ip,
            user_agent=ua,
        )
    )
    db.commit()
