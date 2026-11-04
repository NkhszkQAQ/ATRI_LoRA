"""Lightweight additive schema upgrades for existing Postgres volumes."""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


_STATEMENTS = [
    "ALTER TABLE safety_events ADD COLUMN IF NOT EXISTS ip VARCHAR(64) DEFAULT ''",
    "ALTER TABLE safety_events ADD COLUMN IF NOT EXISTS user_agent TEXT DEFAULT ''",
    "ALTER TABLE safety_events ADD COLUMN IF NOT EXISTS model VARCHAR(120) DEFAULT ''",
    "ALTER TABLE safety_events ADD COLUMN IF NOT EXISTS reply_text TEXT DEFAULT ''",
    "ALTER TABLE safety_events ADD COLUMN IF NOT EXISTS reasoning_text TEXT DEFAULT ''",
    "ALTER TABLE safety_events ADD COLUMN IF NOT EXISTS token_info JSONB",
    "ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS user_agent TEXT DEFAULT ''",
]


def apply_migrations(engine: Engine) -> None:
    with engine.begin() as conn:
        for stmt in _STATEMENTS:
            conn.execute(text(stmt))
