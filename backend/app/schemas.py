from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: EmailStr
    display_name: str
    role: str
    email_verified: bool
    honorific: str
    memory_note: str = ""
    created_at: datetime

    class Config:
        from_attributes = True


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(default="", max_length=120)
    code: str = Field(min_length=4, max_length=16)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class SendCodeIn(BaseModel):
    email: EmailStr
    purpose: Literal["register"] = "register"


class ConversationOut(BaseModel):
    id: int
    title: str
    model: str
    created_at: datetime
    updated_at: datetime
    user_name: Optional[str] = None
    user_email: Optional[str] = None

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    model: str
    meta: Optional[dict[str, Any]] = None
    created_at: datetime
    reasoning: Optional[str] = None

    class Config:
        from_attributes = True


class ChatSendIn(BaseModel):
    conversation_id: Optional[int] = None
    content: str = Field(min_length=1, max_length=8000)
    enable_thinking: bool = True


class AdminUserPatch(BaseModel):
    is_active: Optional[bool] = None
    role: Optional[Literal["user", "admin"]] = None
    display_name: Optional[str] = None
    memory_note: Optional[str] = None
    honorific: Optional[str] = None


class MemoryIn(BaseModel):
    memory_note: str = Field(default="", max_length=4000)


class MemoryOut(BaseModel):
    memory_note: str


class ModelSettingIn(BaseModel):
    model: str


class SmtpSettingIn(BaseModel):
    enabled: bool = False
    host: str = ""
    port: int = 587
    user: str = ""
    password: str = ""
    from_addr: str = ""
    tls: bool = True


class SafetyEventOut(BaseModel):
    id: int
    conversation_id: Optional[int] = None
    category: str
    severity: str
    probability: int
    language_hint: str
    action: str
    raw_text: str
    created_at: datetime
    user_name: str = ""
    user_email: str = ""
    ip: str = ""
    user_agent: str = ""
    model: str = ""
    reply_text: str = ""
    reasoning_text: str = ""
    token_info: Optional[dict[str, Any]] = None
    conversation_title: str = ""

    class Config:
        from_attributes = True


class AuditOut(BaseModel):
    id: int
    actor_name: str = ""
    actor_email: str = ""
    action: str
    target_type: str
    target_id: str
    detail: Optional[dict[str, Any]]
    ip: str
    user_agent: str = ""
    created_at: datetime

    class Config:
        from_attributes = True
