import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atri_bridge import attach_context, detach_context, identity
from atri_memory import MemoryStore, augment_records
from atri_qq_proxy import build_app

SECRET = b"synthetic-local-test-only-secret-123456789"


def test_bridge_signature_and_input_objects_are_preserved():
    rows = [{"key": "偏好饮品", "value": "乌龙茶"}]
    raw = [{"role": "system", "content": attach_context("原人设", "qq:test:123", rows, True, SECRET, now=1000)},
           {"role": "user", "content": "我喜欢喝什么？"}]
    before = json.dumps(raw)
    cleaned, context = detach_context(raw, SECRET, now=1001)
    assert context["records"] == rows and context["owner"] == "qq:test:123"
    assert cleaned[0]["content"] == "原人设" and json.dumps(raw) == before
    with pytest.raises(ValueError):
        detach_context(raw, b"another-secret" * 4, now=1001)
    with pytest.raises(ValueError):
        detach_context(raw, SECRET, now=1200)
    raw[0]["content"] = raw[0]["content"].replace("</ATRI_QQ", "z</ATRI_QQ")
    with pytest.raises(ValueError):
        detach_context(raw, SECRET, now=1001)


def test_group_context_never_includes_private_records():
    content = attach_context("人设", "qq:test:123", [{"key": "口令", "value": "不应出现"}], False, SECRET)
    _, data = detach_context([{"role": "system", "content": content}], SECRET)
    assert data["records"] == []


@pytest.mark.parametrize("value", ["茶" * 400, "\x00" * 400])
def test_full_memory_capacity_fits_signed_transport(value):
    rows = [{"key": "键" * 62 + str(i).zfill(2), "value": value} for i in range(24)]
    text = attach_context("人设", "qq:test:123", rows, True, SECRET)
    _, context = detach_context([{"role": "system", "content": text}], SECRET)
    assert context["records"] == rows


def test_memory_augmentation_preserves_media_parts_and_original_objects():
    messages = [{"role": "user", "content": [{"type": "text", "text": "看看图片"},
                                              {"type": "image_url", "image_url": {"url": "data:image/png;base64,TEST"}}]}]
    original = json.dumps(messages)
    result = augment_records(messages, [{"key": "偏好饮品", "value": "茶"}])
    assert result[-1]["content"][1:] == messages[-1]["content"]
    assert "偏好饮品" in result[-1]["content"][0]["text"] and json.dumps(messages) == original


class Tokenizer:
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def decode(self, tokens, **kwargs): return "".join(chr(c) for c in tokens)


@pytest.fixture
def proxy(tmp_path):
    key = tmp_path / "secret"
    key.write_bytes(SECRET)
    state = {"current": "v1.1-8B", "calls": [], "rag_queries": []}
    cards = json.loads((ROOT / "rag/character_facts.json").read_text())["cards"]
    class RAG:
        grounding_ready = True
        voice_ready = True
        def search(self, question, history=()):
            state["rag_queries"].append(question)
            if "家务" in question:
                card = next(c for c in cards if c["id"] == "housework")
                return [{"metadata": {"kind": "canonical_fact", "answer": card["answer"],
                                       "answer_contract": json.dumps(card["answer_contract"]), "fact_id": card["id"], "spoken_answer": card["spoken_answer"]}}]
            return []
        def augment(self, messages): return messages
    def transport(request):
        if request.url.path == "/models":
            return httpx.Response(200, json={"current": state["current"]})
        body = json.loads(request.content)
        state["calls"].append({"url": str(request.url), "body": body})
        if body.get("stream"):
            return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"stream"}}]}\n\ndata: [DONE]\n\n', headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "我很会做饭。"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8}})
    upstream = httpx.Client(transport=httpx.MockTransport(transport))
    config = {"shared_secret_file": str(key), "gateway": "http://gateway",
              "context_db": str(tmp_path / "context.sqlite3"),
              "backends": {"v1.1-8B": "http://eight", "v1.1-DT": "http://eight", "v1.0-R": "http://four", "v1.0-C": "http://four"}}
    with TestClient(build_app(config, rag=RAG(), client=upstream, tokenizer=Tokenizer())) as client:
        yield client, state
    upstream.close()


def signed_messages(question, *, private=True, rows=None):
    return [{"role": "system", "content": attach_context("保留原人设", "qq:test:123", rows or [], private, SECRET)},
            {"role": "user", "content": question}]


def test_old_fact_index_cannot_start_new_voice_proxy(tmp_path):
    secret = tmp_path / "secret"
    secret.write_bytes(SECRET)
    with pytest.raises(ValueError, match="spoken_answer"):
        build_app({"shared_secret_file": str(secret)}, rag=SimpleNamespace(grounding_ready=True))


@pytest.mark.parametrize("stream", [False, True])
def test_proxy_rejects_bad_fact_drafts_and_keeps_qq_memory_subject(proxy, stream):
    client, state = proxy
    response = client.post("/v1/chat/completions", json={"stream": stream,
        "messages": signed_messages("你会家务吗？另外我喜欢喝什么？", rows=[{"key": "偏好饮品", "value": "乌龙茶"}])})
    assert response.status_code == 200
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        answer = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert response.text.endswith("data: [DONE]\n\n")
    else:
        answer = response.json()["choices"][0]["message"]["content"]
    assert "家务" in answer and "你喜欢喝" in answer and "乌龙茶" in answer
    assert "很会做饭" not in answer and len(state["calls"]) == 2
    assert all("ATRI_QQ_CONTEXT" not in json.dumps(c["body"]) for c in state["calls"])


def test_request_user_field_cannot_select_another_owner_or_group_memory(proxy):
    client, state = proxy
    response = client.post("/v1/chat/completions", json={"user": "bob", "messages": signed_messages(
        "我喜欢喝什么？", rows=[{"key": "偏好饮品", "value": "乌龙茶"}])})
    assert "乌龙茶" in response.json()["choices"][0]["message"]["content"] and not state["calls"]
    response = client.post("/v1/chat/completions", json={"messages": signed_messages("我保存的饮品是什么？", private=False)})
    assert "私聊" in response.json()["choices"][0]["message"]["content"] and not state["calls"]
    response = client.post("/v1/chat/completions", json={"user": "qq:test:123", "messages": [{"role": "user", "content": "我保存的饮品是什么？"}]})
    assert "QQ" in response.json()["choices"][0]["message"]["content"] and not state["calls"]


@pytest.mark.parametrize("model,url", [("v1.1-DT", "http://eight"), ("v1.0-R", "http://four"), ("v1.0-C", "http://four")])
def test_existing_model_switch_is_respected_and_envelope_removed(proxy, model, url):
    client, state = proxy
    state["current"] = model
    response = client.post("/v1/chat/completions", json={"model": "untrusted-client-model", "messages": signed_messages(
        "早安", rows=[{"key": "偏好饮品", "value": "茶"}])})
    assert response.status_code == 200 and not state["rag_queries"]
    call = state["calls"][-1]
    assert call["url"].startswith(url) and call["body"]["model"] == model
    assert "ATRI_QQ_CONTEXT" not in json.dumps(call) and "偏好饮品" in json.dumps(call, ensure_ascii=False)


def test_plain_stream_and_length_limit(proxy):
    client, state = proxy
    response = client.post("/v1/chat/completions", json={"stream": True, "messages": signed_messages("早安")})
    assert "data: [DONE]" in response.text and state["calls"][-1]["body"]["stream"] is True
    response = client.post("/v1/chat/completions", json={"max_tokens": 1, "messages": signed_messages("我保存的饮品是什么？")})
    assert response.json()["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("stream", [False, True])
def test_proxy_bounds_working_history_before_retrieval_and_upstream(proxy, stream):
    client, state = proxy
    messages = signed_messages("那你换一个比喻")
    history = []
    for i in range(12):
        history += [{"role": "user", "content": f"旧话题{i}"}, {"role": "assistant", "content": "旧回复"}]
    messages[1:1] = history
    before = json.dumps(messages)
    response = client.post("/v1/chat/completions", json={"stream": stream, "messages": messages})
    assert response.status_code == 200
    upstream = state["calls"][-1]["body"]["messages"]
    assert "旧话题0" not in str(upstream) and "旧话题4" in str(upstream)
    assert "换一个比喻" in str(upstream) and json.dumps(messages) == before


def test_proxy_idle_boundary_does_not_reappear_on_followup(proxy):
    client, state = proxy
    import time
    now = int(time.time())
    def make(history, sent, previous):
        system = attach_context("原人设", "qq:test:123", [], True, SECRET, turn={
            "conversation_id": "test-cid", "sent_at": sent, "previous_at": previous})
        return [{"role": "system", "content": system}, *history, {"role": "user", "content": "继续"}]
    old = [{"role": "user", "content": "昨天的煽情"}, {"role": "assistant", "content": "睡前回复"}]
    first = client.post("/v1/chat/completions", json={"messages": make(old, now-1, now-90000)})
    assert first.status_code == 200 and "睡前回复" not in str(state["calls"][-1])
    history = old + [{"role": "user", "content": "继续"}, {"role": "assistant", "content": "当前话题"}]
    second = client.post("/v1/chat/completions", json={"messages": make(history, now, now-1)})
    assert second.status_code == 200 and "睡前回复" not in str(state["calls"][-1])
    assert "当前话题" in str(state["calls"][-1])


@pytest.mark.parametrize("bad", [{"conversation_id": "a", "sent_at": float("nan"), "previous_at": 0},
                                 {"conversation_id": "a", "sent_at": 0, "previous_at": 0},
                                 {"conversation_id": "a", "sent_at": 2000, "previous_at": 0},
                                 {"conversation_id": "", "sent_at": 1000, "previous_at": 0}])
def test_signed_conversation_clock_is_validated(bad):
    system = attach_context("原人设", "qq:test:123", [], True, SECRET, now=1000, turn=bad)
    with pytest.raises(ValueError):
        detach_context([{"role": "system", "content": system}], SECRET, now=1001)


def test_plugin_transports_conversation_and_platform_sent_time(plugin):
    async def run():
        import time
        now = int(time.time())
        event = Event("111")
        event.message_obj = SimpleNamespace(timestamp=now-600)
        req = SimpleNamespace(system_prompt="原人设", conversation=SimpleNamespace(cid="test-cid", updated_at=now-610))
        await plugin.memory_context(event, req)
        _, context = detach_context([{"role": "system", "content": req.system_prompt}], SECRET)
        assert context["turn"] == {"conversation_id": "test-cid", "sent_at": now-600, "previous_at": now-610}
    asyncio.run(run())


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    modules = {}
    for name in ["astrbot", "astrbot.api", "astrbot.api.event", "astrbot.core", "astrbot.core.provider", "astrbot.core.provider.entities", "astrbot.core.star", "astrbot.core.star.filter", "astrbot.core.star.filter.command"]:
        modules[name] = ModuleType(name)
    class Star:
        def __init__(self, context): pass
    modules["astrbot.api"].star = SimpleNamespace(Star=Star, Context=object)
    modules["astrbot.api.event"].AstrMessageEvent = object
    modules["astrbot.api.event"].filter = SimpleNamespace(command=lambda *a, **k: lambda f: f, on_llm_request=lambda *a, **k: lambda f: f)
    modules["astrbot.core.provider.entities"].ProviderRequest = object
    modules["astrbot.core.star.filter.command"].GreedyStr = str
    monkeypatch.setitem(sys.modules, "qqtestplugin", ModuleType("qqtestplugin"))
    for name, value in modules.items(): monkeypatch.setitem(sys.modules, name, value)
    folder = tmp_path / "plugin"; folder.mkdir()
    for name in ["atri_bridge.py", "atri_memory.py"]: shutil.copy(ROOT / "tools" / name, folder / name)
    shutil.copy(ROOT / "integrations/astrbot/atri_rag_memory/main.py", folder / "__init__.py")
    secret = tmp_path / "secret"; secret.write_bytes(SECRET)
    config = tmp_path / "runtime.json"; config.write_text(json.dumps({"shared_secret_file": str(secret), "memory_db": str(tmp_path / "memory.sqlite")}))
    (folder / "settings.json").write_text(json.dumps({"config_path": str(config)}))
    spec = importlib.util.spec_from_file_location("qqtestplugin", folder / "__init__.py", submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec); monkeypatch.setitem(sys.modules, "qqtestplugin", module); spec.loader.exec_module(module)
    return module.Main(None)


class Event:
    def __init__(self, sender, private=True): self.sender, self.private, self.stopped = sender, private, False
    def get_platform_id(self): return "test-qq"
    def get_platform_name(self): return "aiocqhttp"
    def get_sender_id(self): return self.sender
    def is_private_chat(self): return self.private
    def plain_result(self, text): return text
    def stop_event(self): self.stopped = True


def test_plugin_only_explicit_private_commands_write_and_users_are_isolated(plugin):
    async def run():
        alice, bob = Event("111"), Event("222")
        answers = [x async for x in plugin.memory(alice, "remember 偏好饮品=乌龙茶")]
        assert "记下啦" in answers[0] and alice.stopped
        answers = [x async for x in plugin.memory(bob, "list")]
        assert answers == ["还没有记下什么呢，主人。"]
        req = SimpleNamespace(system_prompt="原人设", prompt="我现在喜欢红茶")
        await plugin.memory_context(alice, req)
        _, context = detach_context([{"role": "system", "content": req.system_prompt}], SECRET)
        assert context["records"] == [{"key": "偏好饮品", "value": "乌龙茶"}]
        group = Event("111", False)
        assert "私聊" in [x async for x in plugin.memory(group, "clear")][0]
        req = SimpleNamespace(system_prompt="群聊人设")
        await plugin.memory_context(group, req)
        _, context = detach_context([{"role": "system", "content": req.system_prompt}], SECRET)
        assert context["records"] == []
        assert plugin.store.list(identity(alice)) == [{"key": "偏好饮品", "value": "乌龙茶"}]
    asyncio.run(run())
