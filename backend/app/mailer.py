from __future__ import annotations

import smtplib
from email.message import EmailMessage

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import SystemSetting


def _smtp_cfg(db: Session | None = None) -> dict:
    s = get_settings()
    cfg = {
        "enabled": s.smtp_enabled,
        "host": s.smtp_host,
        "port": s.smtp_port,
        "user": s.smtp_user,
        "password": s.smtp_password,
        "from_addr": s.smtp_from or s.smtp_user,
        "tls": s.smtp_tls,
    }
    if db is not None:
        row = db.query(SystemSetting).filter(SystemSetting.key == "smtp").first()
        if row and isinstance(row.value, dict):
            cfg.update({k: row.value.get(k, cfg.get(k)) for k in cfg})
    return cfg


def send_email(to: str, subject: str, body: str, db: Session | None = None) -> bool:
    cfg = _smtp_cfg(db)
    if not cfg.get("enabled"):
        # Dev fallback: print code to logs
        print(f"[smtp-disabled] to={to} subject={subject}\n{body}")
        return True
    msg = EmailMessage()
    msg["From"] = cfg["from_addr"]
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(cfg["host"], int(cfg["port"]), timeout=30) as smtp:
        if cfg.get("tls"):
            smtp.starttls()
        if cfg.get("user"):
            smtp.login(cfg["user"], cfg["password"])
        smtp.send_message(msg)
    return True
