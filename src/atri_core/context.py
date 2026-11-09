"""Select a private QQ working context without rewriting AstrBot's archive."""
from contextlib import closing
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo

LOG = logging.getLogger(__name__)


def fingerprint(messages):
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class ContextWindow:
    def __init__(self, path=None, *, max_history_turns=8, idle_timeout_seconds=21600,
                 timezone="Asia/Tokyo"):
        if (type(max_history_turns) is not int or max_history_turns < 0
                or type(idle_timeout_seconds) is not int or idle_timeout_seconds <= 0):
            raise ValueError("invalid QQ context window limits")
        self.max_turns = max_history_turns
        self.idle_seconds = idle_timeout_seconds
        self.timezone = ZoneInfo(timezone)
        self.path = str(path) if path else None
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            Path(self.path).touch(mode=0o600, exist_ok=True)
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("""CREATE TABLE IF NOT EXISTS context_windows (
                    scope TEXT PRIMARY KEY, last_sent REAL NOT NULL,
                    prefix_count INTEGER NOT NULL, prefix_hash TEXT NOT NULL,
                    cutoff INTEGER NOT NULL)""")

    def _boundary(self, history, owner, turn):
        if not self.path or not turn:
            return 0, None, False
        scope = fingerprint([owner, turn["conversation_id"]])
        sent = turn["sent_at"]
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT last_sent, prefix_count, prefix_hash, cutoff "
                             "FROM context_windows WHERE scope=?", (scope,)).fetchone()
            cutoff, previous = 0, turn["previous_at"] or None
            if row:
                last, count, digest, saved_cutoff = row
                # Reset, editing or native compression invalidates numeric offsets.
                if count <= len(history) and fingerprint(history[:count]) == digest:
                    cutoff, previous = saved_cutoff, last
                else:
                    previous = max(last, previous or 0)
                sent = max(sent, last)  # delayed/duplicate events cannot move time back
            gap = max(0, sent - previous) if previous is not None else None
            expired = gap is not None and gap >= self.idle_seconds
            if expired:
                cutoff = len(history)
            db.execute("INSERT INTO context_windows VALUES (?, ?, ?, ?, ?) "
                       "ON CONFLICT(scope) DO UPDATE SET last_sent=excluded.last_sent, "
                       "prefix_count=excluded.prefix_count, prefix_hash=excluded.prefix_hash, "
                       "cutoff=excluded.cutoff",
                       (scope, sent, len(history), fingerprint(history), cutoff))
        return cutoff, gap, expired

    def select(self, messages, context, *, now=None):
        if not context or not context["private"]:
            return messages, None
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not users:
            return messages, None
        current = users[-1]
        history = [m for m in messages[:current] if m.get("role") != "system"]
        cutoff, gap, expired = self._boundary(history, context["owner"], context.get("turn"))
        starts = [i for i, m in enumerate(history) if m.get("role") == "user" and i >= cutoff]
        if not self.max_turns or not starts:
            start = len(history)
        else:
            start = starts[-self.max_turns] if len(starts) > self.max_turns else starts[0]
        # A user turn includes all following assistant/tool messages, atomically.
        result = [dict(m) for m in messages if m.get("role") == "system"]
        result.extend(dict(m) for m in history[start:])
        result.extend(dict(m) for m in messages[current:] if m.get("role") != "system")
        clock = time.time() if now is None else now
        sent = context.get("turn", {}).get("sent_at", clock)
        reminder = ("\n[当前会话边界]\n当前时间：" + datetime.fromtimestamp(clock, self.timezone).isoformat(timespec="seconds")
                    + "；本条消息发送时间：" + datetime.fromtimestamp(sent, self.timezone).isoformat(timespec="seconds")
                    + "。优先回应最新消息，历史仅用于理解当前话题；用户要求改写或换例子时，直接给出新的内容。"
                    "不要无关地续接以前的调侃、争论或睡前话题。")
        if expired:
            reminder += "距离前次消息已超过会话间隔，当前开始新的短期话题。"
        system = next((m for m in result if m.get("role") == "system" and isinstance(m.get("content"), str)), None)
        if system is None:
            result.insert(0, {"role": "system", "content": reminder.strip()})
        else:
            system["content"] += reminder
        report = {"history_messages": len(history), "kept_messages": len(history) - start,
                  "idle_boundary": expired, "gap_seconds": gap}
        LOG.info("QQ working context: history=%s kept=%s idle_boundary=%s",
                 report["history_messages"], report["kept_messages"], expired)
        return result, report
