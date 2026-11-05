import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atri_rag import Config, Retriever, build_index, from_env, read_documents
from llama_index.core.embeddings import BaseEmbedding


@pytest.mark.parametrize("question,history,expected", [
    ("你尝得出味道吗？", [], "亚托莉尝得出味道吗？"),
    ("她是人类吗？", [], "亚托莉是人类吗？"),
    ("战斗型号已确认吗？", [], "亚托莉：战斗型号已确认吗？"),
    ("夏生的腿怎么了？", [], "夏生的腿怎么了？"),
    ("她想上学吗？", [{"role": "user", "content": "说说水菜萌"}], "水菜萌想上学吗？"),
    ("她是谁？", [{"role": "user", "content": "亚托莉和水菜萌"}], None),
    ("她是谁？", [{"role": "user", "content": "我的姐姐昨天来过"}], None),
    ("你有味觉吗？", [{"role": "user", "content": "水菜萌"}], "亚托莉有味觉吗？"),
    ("她是谁？", [{"role": "assistant", "content": "水菜萌"}], "亚托莉是谁？"),
    ("她是谁？", [{"role": "user", "content": "/memory remember name=水菜萌"}], "亚托莉是谁？"),
    ("你会烹饪吗？", [], "亚托莉会烹饪等家务吗？"),
    ("亚托莉擅长做饭吗？", [], "亚托莉擅长做饭等家务吗？"),
    ("亚托莉喜欢什么食物？", [], "亚托莉喜欢什么食物？"),
])
def test_bounded_query_rewriting(question, history, expected):
    from atri_rag import query_variants
    original = json.dumps(history)
    queries = query_variants(question, history)
    if expected is None:
        assert queries == []
    else:
        assert queries[0] == question
        assert queries[-1] == expected
    assert len(queries) <= 2
    assert json.dumps(history) == original


def test_reranking_deduplicates_and_preserves_provenance(corpus, tmp_path):
    storage = tmp_path / "index"
    cfg = Config(rerank_enabled=True)
    build_index(corpus, storage, cfg, embed_model=TopicEmbedding())
    class Ranker:
        calls = []
        def score(self, query, texts):
            self.calls.append((query, list(texts)))
            return [0.9 if "学校" in text else 0.1 for text in texts]
    ranker = Ranker()
    rag = Retriever(storage, cfg, embed_model=TopicEmbedding(), reranker=ranker)
    hits = rag.search("你知道学校在哪里吗？")
    assert len(hits) == 1
    assert hits[0]["score"] == 0.9
    assert hits[0]["metadata"]["source"] == "story.txt"
    assert hits[0]["metadata"]["scene"] == 1
    assert "retrieval_score" in hits[0] and "retrieval_query" in hits[0]
    assert ranker.calls[-1][0] == "亚托莉知道学校在哪里吗？"
    assert len(ranker.calls[-1][1]) == len(set(ranker.calls[-1][1]))
    # Filtering everything must leave the user's message intact.
    ranker.score = lambda q, texts: [0.0] * len(texts)
    messages = [{"role": "user", "content": "学校在哪里？"}]
    assert rag.augment(messages) == messages


@pytest.mark.parametrize("legacy", [False, True])
def test_local_reranker_normalizes_logits(monkeypatch, legacy):
    import sentence_transformers
    import torch
    from atri_rag import LocalReranker
    class Modern:
        def __init__(self, model, **kwargs):
            assert kwargs["device"] == "cpu" and kwargs["trust_remote_code"] is False
        def predict(self, pairs, activation_fn=None, **kwargs):
            return activation_fn(torch.tensor([0.0, 2.0])).numpy()
    class Legacy(Modern):
        def predict(self, pairs, activation_fct=None, **kwargs):
            return activation_fct(torch.tensor([0.0, 2.0])).numpy()
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Legacy if legacy else Modern)
    scores = LocalReranker("test-fixture").score("问题", ["片段1", "片段2"])
    assert scores[0] == 0.5
    assert 0.88 < scores[1] < 0.89


def test_supplementary_sources_and_citation_validation(corpus, tmp_path):
    import hashlib
    from atri_rag import supplementary_documents
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "persona_card.md").write_text("- 不会：吃东西（旧设定）\n称呼用户为主人")
    (persona / "character_profile.md").write_text("可以品尝味道")
    card = {"id": "school", "title": "学校", "questions": ["学校在哪"],
            "answer": "学校在山坡上", "scope": "early_story", "evidence": [{
                "source": "story.txt", "sha256": hashlib.sha256((corpus / "story.txt").read_bytes()).hexdigest(),
                "scene": 1, "dialogue_start": 1, "dialogue_end": 1}]}
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"version": 1, "cards": [card]}))
    docs, _ = supplementary_documents(corpus, persona, facts)
    assert len(docs) == 3
    assert docs[-1].metadata["fact_id"] == "school"
    assert json.loads(docs[-1].metadata["evidence"])[0]["scene"] == 1
    assert "不能摄取营养" in docs[0].text
    assert "旧设定" in (persona / "persona_card.md").read_text()
    card["evidence"][0]["sha256"] = "changed"
    facts.write_text(json.dumps({"version": 1, "cards": [card]}))
    with pytest.raises(ValueError, match="hash mismatch"):
        supplementary_documents(corpus, persona, facts)


def test_cutoff_and_greetings_do_not_inject_weak_matches():
    import threading
    from llama_index.core.schema import NodeWithScore, TextNode
    rag = object.__new__(Retriever)
    rag.config = Config(rerank_enabled=False, min_score=0.6)
    rag.lock = threading.Lock()
    rag.reranker = None
    rag.fact_retriever = None
    class Backend:
        def retrieve(self, query):
            return [NodeWithScore(node=TextNode(text="不相关片段"), score=0.4)]
    rag.retriever = Backend()
    for query in ("你好呀！", "今天东京天气如何", ""):
        messages = [{"role": "user", "content": query}]
        assert rag.search(query) == []
        assert rag.augment(messages) == messages


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
    cfg = Config(rerank_enabled=False, top_k=1)
    manifest = build_index(corpus, storage, cfg, embed_model=TopicEmbedding())
    assert manifest["documents"] == 3
    assert manifest["nodes"] >= 3
    retrieved = Retriever(storage, cfg, embed_model=TopicEmbedding())
    assert retrieved.grounding_ready is False
    hit = retrieved.search("学校")[0]
    assert hit["metadata"]["source"] == "story.txt"
    assert hit["metadata"]["scene"] == 1
    assert hit["node_id"] and hit["metadata"]["end_char"] > 0
    assert retrieved.search("  ") == []
    with pytest.raises(ValueError, match="mismatch"):
        Retriever(storage, Config(rerank_enabled=False, model="different"), embed_model=TopicEmbedding())
    before = (storage / "atri_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        build_index(corpus, storage, cfg, embed_model=TopicEmbedding())
    assert before == (storage / "atri_manifest.json").read_bytes()


def test_grounding_readiness_is_checked_against_persisted_fact_metadata(corpus, tmp_path):
    import hashlib
    storage = tmp_path / "index"
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"version": 1, "cards": [{
        "id": "school", "title": "学校", "questions": ["学校在哪"],
        "answer": "学校在山坡上", "scope": "early_story", "evidence": [{
            "source": "story.txt", "sha256": hashlib.sha256((corpus / "story.txt").read_bytes()).hexdigest(),
            "scene": 1, "dialogue_start": 1, "dialogue_end": 1}]}]}))
    cfg = Config(rerank_enabled=False)
    build_index(corpus, storage, cfg, facts=facts, embed_model=TopicEmbedding())
    assert Retriever(storage, cfg, embed_model=TopicEmbedding()).grounding_ready is True
    # Older snapshots had fact IDs and citations, but no complete answer metadata.
    path = storage / "docstore.json"
    data = json.loads(path.read_text())
    def remove_old_missing_field(value):
        if isinstance(value, dict):
            if value.get("kind") == "canonical_fact":
                value.pop("answer", None)
            for child in value.values():
                remove_old_missing_field(child)
        elif isinstance(value, list):
            for child in value:
                remove_old_missing_field(child)
    remove_old_missing_field(data)
    path.write_text(json.dumps(data))
    assert Retriever(storage, cfg, embed_model=TopicEmbedding()).grounding_ready is False


def test_augmentation_preserves_history_and_question(corpus, tmp_path):
    storage = tmp_path / "index"
    cfg = Config(rerank_enabled=False, top_k=1)
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
    rag.config = Config(rerank_enabled=False, max_context_chars=300)
    rag.search = lambda q, *args: [{"node_id": "1", "score": 1, "metadata": {},
                             "text": '\\"学校\n' * 1000}]
    msgs = [{"role": "user", "content": "问题"}]
    result = rag.augment(msgs)
    context = result[-1]["content"].split("参考资料（JSON 数据）：\n")[1].split("\n\n用户问题：")[0]
    assert len(context) <= 300
    assert json.loads(context)[0]["node_id"] == "1"
    rag.search = lambda q, *args: []
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


@pytest.mark.parametrize("memory_enabled", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_server_plain_and_sse(monkeypatch, tmp_path, enabled, stream, memory_enabled):
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

    queries = []
    class RAG:
        def augment(self, msgs):
            queries.append(msgs[-1]["content"])
            return [*msgs[:-1], {**msgs[-1], "content": "检索上下文：" + msgs[-1]["content"]}]

    monkeypatch.setattr("atri_rag.from_env", lambda: RAG() if enabled else None)
    monkeypatch.setenv("ATRI_RAG_ENABLED", "1" if enabled else "0")
    monkeypatch.setattr(transformers, "TextIteratorStreamer", Streamer)
    monkeypatch.setenv("ATRI_MEMORY_ENABLED", "1" if memory_enabled else "0")
    token = "test-only-token-" * 4
    if memory_enabled:
        from atri_memory import MemoryStore
        credentials = tmp_path / "users.json"
        credentials.write_text(json.dumps({"alice": token, "bob": "bob-test-only-token-" * 4}))
        monkeypatch.setenv("ATRI_MEMORY_USERS_FILE", str(credentials))
        monkeypatch.setenv("ATRI_MEMORY_DB", str(tmp_path / "state.sqlite3"))
        MemoryStore(tmp_path / "state.sqlite3").put("alice", "name", "小明")
        MemoryStore(tmp_path / "state.sqlite3").put("bob", "name", "小红")
    tok = Tokenizer()
    monkeypatch.setattr(server, "state", {"tok": tok, "model": Model()})
    monkeypatch.setattr(server, "default_system", lambda: "ATRI 人设")
    with TestClient(server.build_app()) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + token}, json={"messages": [
            {"role": "user", "content": [{"type": "text", "text": "学校？"}]}],
            "stream": stream, "max_tokens": None})
        assert response.status_code == 200
        assert tok.seen[0]["content"].startswith("ATRI 人设")
        expected = "检索上下文：学校？" if enabled else "学校？"
        assert tok.seen[-1]["content"].endswith(expected)
        assert queries == (["学校？"] if enabled else [])
        if memory_enabled:
            assert "小明" in tok.seen[-1]["content"]
            assert "小红" not in tok.seen[-1]["content"]
            denied = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "学校？"}]})
            assert denied.status_code == 401
            assert queries == (["学校？"] if enabled else [])
            # Memory commands must bypass RAG as well as generation.
            client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + token},
                        json={"messages": [{"role": "user", "content": "/memory list"}]})
            assert queries == (["学校？"] if enabled else [])
        if stream:
            chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            assert chunks[0]["choices"][0]["delta"]["content"] == "测试回答"
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            assert response.json()["choices"][0]["message"]["content"] == "测试回答"
