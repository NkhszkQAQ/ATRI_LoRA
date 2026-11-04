from __future__ import annotations

import getpass
import os
from typing import Optional

import typer
from rich import print as rprint

from app.config import get_settings
from app.db import Base, SessionLocal, engine
from app.models import SystemSetting, User, UserRole
from app.security import hash_password

cli = typer.Typer(help="ATRI LoRA 管理命令")


@cli.command("init")
def init_cmd(
    admin_email: Optional[str] = typer.Option(None, help="管理员邮箱"),
    admin_password: Optional[str] = typer.Option(None, help="管理员密码"),
    admin_name: Optional[str] = typer.Option(None, help="显示名"),
    smtp_enabled: Optional[bool] = typer.Option(None, help="是否启用 SMTP"),
    yes: bool = typer.Option(False, "--yes", "-y", help="非交互，使用环境变量"),
):
    """初始化数据库、首个管理员、可选 SMTP。"""
    Base.metadata.create_all(bind=engine)
    settings = get_settings()
    db = SessionLocal()
    try:
        email = admin_email or os.getenv("ADMIN_EMAIL") or settings.__dict__.get("admin_email")
        if not email:
            email = "admin@example.com" if yes else typer.prompt("管理员邮箱", default=os.getenv("ADMIN_EMAIL", "admin@example.com"))
        name = admin_name or os.getenv("ADMIN_DISPLAY_NAME", "Admin")
        if not yes and not admin_name:
            name = typer.prompt("管理员显示名", default=name)

        password = admin_password or os.getenv("ADMIN_PASSWORD")
        if not password:
            if yes:
                raise typer.BadParameter("非交互模式需要 ADMIN_PASSWORD 或 --admin-password")
            password = getpass.getpass("管理员密码: ")
            password2 = getpass.getpass("确认密码: ")
            if password != password2:
                raise typer.Exit("两次密码不一致")

        existing = db.query(User).filter(User.email == email).first()
        if existing:
            existing.role = UserRole.admin
            existing.password_hash = hash_password(password)
            existing.display_name = name
            existing.email_verified = True
            existing.is_active = True
            rprint(f"[yellow]已更新现有管理员[/yellow] {email}")
        else:
            db.add(
                User(
                    email=email,
                    password_hash=hash_password(password),
                    display_name=name,
                    role=UserRole.admin,
                    email_verified=True,
                )
            )
            rprint(f"[green]已创建管理员[/green] {email}")

        # model default
        if not db.query(SystemSetting).filter(SystemSetting.key == "model").first():
            db.add(SystemSetting(key="model", value={"model": settings.atri_model}))

        # smtp
        enable = smtp_enabled
        if enable is None:
            if yes:
                enable = settings.smtp_enabled
            else:
                enable = typer.confirm("启用 SMTP 邮箱验证？", default=settings.smtp_enabled)

        smtp_val = {
            "enabled": bool(enable),
            "host": settings.smtp_host,
            "port": settings.smtp_port,
            "user": settings.smtp_user,
            "password": settings.smtp_password,
            "from_addr": settings.smtp_from or settings.smtp_user,
            "tls": settings.smtp_tls,
        }
        if enable and not yes:
            smtp_val["host"] = typer.prompt("SMTP Host", default=smtp_val["host"] or "")
            smtp_val["port"] = int(typer.prompt("SMTP Port", default=str(smtp_val["port"])))
            smtp_val["user"] = typer.prompt("SMTP User", default=smtp_val["user"] or "")
            smtp_val["password"] = getpass.getpass("SMTP Password: ") or smtp_val["password"]
            smtp_val["from_addr"] = typer.prompt("From", default=smtp_val["from_addr"] or smtp_val["user"])
            smtp_val["tls"] = typer.confirm("使用 STARTTLS？", default=True)

        row = db.query(SystemSetting).filter(SystemSetting.key == "smtp").first()
        if not row:
            db.add(SystemSetting(key="smtp", value=smtp_val))
        else:
            row.value = smtp_val

        db.commit()
        rprint("[bold green]初始化完成[/bold green]")
        rprint(f"默认模型: {settings.atri_model}")
        rprint(f"SMTP: {'启用' if smtp_val['enabled'] else '关闭（验证码打印到 API 日志）'}")
    finally:
        db.close()


@cli.command("reset-admin-password")
def reset_admin_password(email: str, password: str):
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()
        if not u:
            raise typer.Exit("用户不存在")
        u.password_hash = hash_password(password)
        u.role = UserRole.admin
        db.commit()
        rprint("[green]密码已重置[/green]")
    finally:
        db.close()


if __name__ == "__main__":
    cli()
