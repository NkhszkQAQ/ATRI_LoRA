from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        default="postgresql+psycopg://atri:change-me-atri-db@localhost:5433/atri",
        alias="DATABASE_URL",
    )
    secret_key: str = Field(default="dev-secret-change-me", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=60 * 24 * 7, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    cors_origins: str = Field(default="http://localhost:3080", alias="CORS_ORIGINS")

    ollama_base_url: str = Field(default="http://host.docker.internal:8008", alias="OLLAMA_BASE_URL")
    atri_model: str = Field(default="atri-qwen3-4b-lora", alias="ATRI_MODEL")

    smtp_enabled: bool = Field(default=False, alias="SMTP_ENABLED")
    smtp_host: str = Field(default="", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_from: str = Field(default="", alias="SMTP_FROM")
    smtp_tls: bool = Field(default=True, alias="SMTP_TLS")

    persona_card_path: str = Field(default="persona/persona_card.md", alias="PERSONA_CARD_PATH")
    system_prompt_path: str = Field(default="persona/system_prompt.txt", alias="SYSTEM_PROMPT_PATH")

    @property
    def cors_origin_list(self) -> List[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
