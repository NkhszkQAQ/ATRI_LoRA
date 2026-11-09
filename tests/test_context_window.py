import copy
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atri_context import ContextWindow

NOW = 1793934000


def context(sent=NOW, previous=0, *, cid="conversation-a", owner="qq:test:111"):
    return {"owner": owner, "private": True, "records": [], "turn": {
        "conversation_id": cid, "sent_at": sent, "previous_at": previous}}


def pair(question, answer="回复"):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def question(history, text="继续"):
    return [{"role": "system", "content": "原人设"}, *history, {"role": "user", "content": text}]


def test_screenshot_history_is_excluded_but_current_analogy_is_kept():
    history = pair("你好煽情", "不是煽情")
    for i in range(6):
        history += pair(f"其他话题{i}")
    history += pair("你咋做", "就像调肉馅") + pair("这是啥比喻啊")
    original = question(history, "那你换一个比喻")
    before = copy.deepcopy(original)
    selected, report = ContextWindow().select(original, context(), now=NOW)
    encoded = json.dumps(selected, ensure_ascii=False)
    assert "你好煽情" not in encoded and "不是煽情" not in encoded
    assert "就像调肉馅" in encoded and selected[-1]["content"] == "那你换一个比喻"
    assert report["kept_messages"] == 16 and original == before


def test_idle_boundary_survives_next_turn_and_proxy_restart(tmp_path):
    db = tmp_path / "context.sqlite3"
    history = pair("昨晚睡前话题")
    selected, report = ContextWindow(db).select(question(history, "今天报名"), context(previous=NOW-90000), now=NOW)
    assert report["idle_boundary"] and "昨晚睡前话题" not in str(selected)
    history += pair("今天报名", "先查截止时间")
    selected, report = ContextWindow(db).select(question(history, "你咋做"), context(NOW+10), now=NOW+10)
    assert "昨晚睡前话题" not in str(selected) and "先查截止时间" in str(selected)
    assert not report["idle_boundary"]
    with sqlite3.connect(db) as connection:
        stored = connection.execute("SELECT * FROM context_windows").fetchall()
    assert "睡前" not in str(stored) and "qq:test:111" not in str(stored)


def test_window_does_not_split_a_turns_tool_calls_or_media():
    image = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:TEST"}}]}
    calls = [{"role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "call1", "content": "result"},
             {"role": "assistant", "content": "查到了"}]
    selected, _ = ContextWindow(max_history_turns=1).select(question(pair("旧话题") + [image, *calls]), context(), now=NOW)
    assert selected[1:-1] == [image, *calls]


@pytest.mark.parametrize("private", [False, None])
def test_groups_and_unsigned_calls_keep_the_original_context(private):
    messages = question(pair("原历史"))
    ctx = None if private is None else {"private": private}
    selected, report = ContextWindow(max_history_turns=0).select(messages, ctx)
    assert selected is messages and report is None


def test_owner_conversation_and_resets_do_not_share_boundaries(tmp_path):
    window = ContextWindow(tmp_path / "state.db")
    old = pair("旧话题")
    window.select(question(old), context(previous=NOW-90000), now=NOW)
    for ctx in [context(cid="conversation-b"), context(owner="qq:test:222")]:
        selected, _ = window.select(question(old), ctx, now=NOW)
        assert "旧话题" in str(selected)
    edited = pair("/reset 后的新内容")
    selected, _ = window.select(question(edited), context(NOW+1), now=NOW+1)
    assert "/reset 后的新内容" in str(selected)
    selected, _ = window.select(question([]), context(NOW+2), now=NOW+2)
    assert selected[-1]["content"] == "继续"


def test_event_timestamp_not_processing_delay_sets_the_idle_boundary(tmp_path):
    window = ContextWindow(tmp_path / "state.db", idle_timeout_seconds=100)
    history = pair("刚才的话题")
    selected, report = window.select(question(history), context(sent=NOW-200, previous=NOW-210), now=NOW)
    assert not report["idle_boundary"] and "刚才的话题" in str(selected)
    assert "2026-" in selected[0]["content"] and "+09:00" in selected[0]["content"]
    # A delayed event cannot roll a scope's persisted clock backwards.
    window.select(question(history), context(sent=NOW-205), now=NOW)
    selected, report = window.select(question(history), context(sent=NOW-105), now=NOW)
    assert not report["idle_boundary"]


def test_clock_continues_from_existing_state_without_bootstrap_timestamp(tmp_path):
    window = ContextWindow(tmp_path / "state.db", idle_timeout_seconds=100)
    history = pair("旧话题")
    window.select(question(history), context(), now=NOW)
    selected, report = window.select(question(history + pair("继续")), context(NOW+100), now=NOW+100)
    assert report["idle_boundary"] and "旧话题" not in str(selected)


@pytest.mark.parametrize("options", [{"max_history_turns": -1}, {"max_history_turns": True}, {"idle_timeout_seconds": 0}])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ContextWindow(**options)
