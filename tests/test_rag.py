import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atri_rag import Config, Retriever, build_index, from_env, read_documents
from llama_index.core.embeddings import BaseEmbedding


class TopicEmbedding(BaseEmbedding):
    """Offline fixture: test retrieval plumbing, not semantic model quality."""
    def _get_text_embedding(self, text):
        return [float(text.count(word)) + 0.01 for word in ("学校", "螃蟹", "电池")]

    def _get_query_embedding(self, query):
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query):
        return self._get_query_embedding(query)


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    script = [{"dialogues": [{"character": "测试角色", "text": "学校在山坡上，学校有教室。"}]},
              {"dialogues": [{"character": "", "text": "海边有螃蟹，螃蟹住在沙滩。"}]}]
    (source / "story.txt").write_text(json.dumps(script, ensure_ascii=False))
    (source / "notes.md").write_text("电池需要充电，电池提供能量。")
    return source


def test_script_provenance_and_plain_text(corpus):
    docs, inventory = read_documents(corpus)
    assert len(docs) == 3
    school = next(d for d in docs if d.metadata.get("scene") == 1)
    assert "[1] 测试角色：学校" in school.text
    assert school.metadata["source"] == "story.txt"
    assert school.metadata["entry_count"] == 1
    assert len(school.metadata["sha256"]) == 64
    assert len(inventory) == 2
    assert all("/Users/" not in json.dumps(d.metadata) for d in docs)


def test_hidden_and_symlink_files_are_not_imported(corpus, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("秘密")
    (corpus / "linked.txt").symlink_to(secret)
    (corpus / ".hidden.txt").write_text("秘密")
    docs, inventory = read_documents(corpus)
    assert len(inventory) == 2
    assert all("秘密" not in doc.text for doc in docs)


def test_empty_and_invalid_sources_fail(tmp_path):
    with pytest.raises(ValueError, match="no nonempty"):
        read_documents(tmp_path)
    (tmp_path / "bad.json").write_text('{"password": "not a script"}')
    with pytest.raises(ValueError, match="scene array"):
        read_documents(tmp_path)


def test_branch_choices_preserve_scene_numbers(tmp_path):
    (tmp_path / "branch.txt").write_text(json.dumps([
        {"choices": [{"text": "去学校", "nextScene": 3}]},
        {"dialogues": [{"character": "测试", "text": "螃蟹"}]}]))
    docs, _ = read_documents(tmp_path)
    assert docs[0].metadata["scene_type"] == "choices"
    assert "不代表已经发生" in docs[0].text
    assert docs[1].metadata["scene"] == 2


def test_persistence_search_and_mismatch(corpus, tmp_path):
    storage = tmp_path / "index"
    cfg = Config(top_k=1)
    manifest = build_index(corpus, storage, cfg, embed_model=TopicEmbedding())
    assert manifest["documents"] == 3
    assert manifest["nodes"] >= 3
    retrieved = Retriever(storage, cfg, embed_model=TopicEmbedding())
    hit = retrieved.search("学校")[0]
    assert hit["metadata"]["source"] == "story.txt"
    assert hit["metadata"]["scene"] == 1
    assert hit["node_id"] and hit["metadata"]["end_char"] > 0
    assert retrieved.search("  ") == []
    with pytest.raises(ValueError, match="mismatch"):
        Retriever(storage, Config(model="different"), embed_model=TopicEmbedding())
    before = (storage / "atri_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        build_index(corpus, storage, cfg, embed_model=TopicEmbedding())
    assert before == (storage / "atri_manifest.json").read_bytes()


def test_augmentation_preserves_history_and_question(corpus, tmp_path):
    storage = tmp_path / "index"
    cfg = Config(top_k=1)
    build_index(corpus, storage, cfg, embed_model=TopicEmbedding())
    rag = Retriever(storage, cfg, embed_model=TopicEmbedding())
    original = [{"role": "system", "content": "你是 ATRI。"},
                {"role": "assistant", "content": "你好"},
                {"role": "user", "content": "学校在哪里？"}]
    baseline = json.dumps(original)
    result = rag.augment(original)
    assert json.dumps(original) == baseline
    assert result[0]["content"].startswith("你是 ATRI。")
    assert result[1] == original[1]
    assert result[-1]["content"].endswith("用户问题：\n学校在哪里？")
    assert '"scene": 1' in result[-1]["content"]
    assert rag.augment(original[:-1]) == original[:-1]


def test_context_budget_and_empty_retrieval():
    rag = object.__new__(Retriever)
    rag.config = Config(max_context_chars=300)
    rag.search = lambda q: [{"node_id": "1", "score": 1, "metadata": {},
                             "text": '\\"学校\n' * 1000}]
    msgs = [{"role": "user", "content": "问题"}]
    result = rag.augment(msgs)
    context = result[-1]["content"].split("参考资料（JSON 数据）：\n")[1].split("\n\n用户问题：")[0]
    assert len(context) <= 300
    assert json.loads(context)[0]["node_id"] == "1"
    rag.search = lambda q: []
    assert rag.augment(msgs) == msgs


def test_disabled_requires_no_llama_dependencies(monkeypatch):
    monkeypatch.delenv("ATRI_RAG_ENABLED", raising=False)
    with patch("atri_rag.Retriever", side_effect=AssertionError("must not load")):
        assert from_env() is None
    monkeypatch.setenv("ATRI_RAG_ENABLED", "1")
    monkeypatch.delenv("ATRI_RAG_STORAGE", raising=False)
    with pytest.raises(ValueError, match="STORAGE"):
        from_env()


def test_disabled_server_needs_no_companion_module(monkeypatch):
    import serve_v1_1_8b as server
    monkeypatch.setenv("ATRI_RAG_ENABLED", "0")
    monkeypatch.setitem(sys.modules, "atri_rag", None)
    assert server.build_app() is not None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_server_plain_and_sse(monkeypatch, enabled, stream):
    """Real HTTP handler + real torch, with generation replaced by a tiny fixture."""
    import queue
    import torch
    import transformers
    from fastapi.testclient import TestClient
    import serve_v1_1_8b as server

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1
        seen = None

        def apply_chat_template(self, msgs, **kwargs):
            self.seen = msgs
            return "prompt"

        def __call__(self, *args, **kwargs):
            return Inputs(input_ids=torch.tensor([[1, 2]]))

        def decode(self, *args, **kwargs):
            return "测试回答"

    class Streamer:
        def __init__(self, *args, **kwargs):
            self.q = queue.Queue()

        def __iter__(self):
            while (item := self.q.get(timeout=5)) is not None:
                yield item

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            if "streamer" in kwargs:
                kwargs["streamer"].q.put("测试回答")
                kwargs["streamer"].q.put(None)
            return torch.tensor([[1, 2, 3]])

    class RAG:
        def augment(self, msgs):
            return [*msgs[:-1], {**msgs[-1], "content": "检索上下文：" + msgs[-1]["content"]}]

    monkeypatch.setattr("atri_rag.from_env", lambda: RAG() if enabled else None)
    monkeypatch.setenv("ATRI_RAG_ENABLED", "1" if enabled else "0")
    monkeypatch.setattr(transformers, "TextIteratorStreamer", Streamer)
    tok = Tokenizer()
    monkeypatch.setattr(server, "state", {"tok": tok, "model": Model()})
    monkeypatch.setattr(server, "default_system", lambda: "ATRI 人设")
    with TestClient(server.build_app()) as client:
        response = client.post("/v1/chat/completions", json={"messages": [
            {"role": "user", "content": [{"type": "text", "text": "学校？"}]}],
            "stream": stream, "max_tokens": None})
        assert response.status_code == 200
        assert tok.seen[0]["content"] == "ATRI 人设"
        assert tok.seen[-1]["content"] == ("检索上下文：学校？" if enabled else "学校？")
        if stream:
            chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            assert chunks[0]["choices"][0]["delta"]["content"] == "测试回答"
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            assert response.json()["choices"][0]["message"]["content"] == "测试回答"
