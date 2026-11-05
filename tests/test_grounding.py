import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atri_grounding import AnswerTask, check_answer, make_plan, render_plan, memory_reference, split_questions


@pytest.mark.parametrize("text,context,expected", [
    ("我是陈甲先生的助手。", "你是谁？", "我是主人的助手。"),
    ("李乙先生，早安。", "早安", "主人，早安。"),
    ("夏生先生的腿受伤了。", "夏生的腿怎么了", "夏生先生的腿受伤了。"),
    ("王先生，您好。", "请给王先生写一封信", "王先生，您好。"),
    ("阿托莉有味觉。", "亚托莉", "亚托莉有味觉。"),
])
def test_generated_name_cleanup_preserves_names_provided_in_context(text, context, expected):
    from atri_grounding import normalize_output
    assert normalize_output(text, context) == expected


def test_old_index_is_not_silently_treated_as_v13_grounded_data(monkeypatch):
    import serve_v1_1_8b as server
    class OldIndex:
        grounding_ready = False
    monkeypatch.setenv("ATRI_RAG_ENABLED", "1")
    monkeypatch.setenv("ATRI_GROUNDING_ENABLED", "1")
    monkeypatch.setattr("atri_rag.from_env", lambda: OldIndex())
    with pytest.raises(ValueError, match="rebuilt with --facts"):
        server.build_app()

CARDS = json.loads((Path(__file__).resolve().parents[1] / "rag/character_facts.json").read_text())["cards"]


@pytest.mark.parametrize("card", CARDS, ids=lambda card: card["id"])
def test_authored_references_satisfy_their_contracts(card):
    assert check_answer(card["answer"], AnswerTask("核对", "fact", card["answer"], card["answer_contract"])) == []


@pytest.mark.parametrize("fact_id,draft", [
    ("housework", "她确实很会做饭，已经学会做饭了。"),
    ("school", "只是想学习知识，至于其他原因她没说。"),
    ("taste", "尝不出味道，但我可以描述味道。"),
    ("identity", "我是真正的人类女孩。"),
    ("taste", "我不能尝味道，不过有味觉的描述功能。"),
    ("eating", "我不能吃饭，也不能补充营养。"),
    ("school", "她想学习知识，不是为了寻找记忆。"),
])
def test_previous_real_model_failures_are_not_sent(fact_id, draft):
    card = next(c for c in CARDS if c["id"] == fact_id)
    task = AnswerTask("事实问题", "fact", card["answer"], card["answer_contract"])
    calls = []
    def generate(messages):
        calls.append(messages)
        return draft
    answer, trace = render_plan([task], "ATRI人设", generate)
    assert len(calls) == 2
    assert card["answer"] in answer
    assert trace[0]["method"] == "source_fallback"
    assert draft not in answer


def test_good_draft_and_repair_are_used_before_fallback():
    card = next(c for c in CARDS if c["id"] == "taste")
    task = AnswerTask("你有味觉吗", "fact", card["answer"], card["answer_contract"])
    responses = iter(["我尝不出。", "我有味觉，能尝味道。"])
    answer, trace = render_plan([task], "ATRI人设", lambda msgs: next(responses))
    assert answer == "我有味觉，能尝味道。"
    assert trace[0]["method"] == "repair"


def test_unknown_fact_and_memory_are_not_invented_by_model():
    class RAG:
        def search(self, *args):
            return []
    tasks = make_plan([{"role": "user", "content": "亚托莉要用什么充电器？另外我保存的代号是什么？"}], RAG(), [])
    def forbidden(messages):
        pytest.fail("unknown facts / missing saved memory must not ask model to guess")
    answer, trace = render_plan(tasks, "ATRI人设", forbidden)
    assert "不能确认" in answer and "没有找到" in answer
    assert len(trace) == 2


@pytest.mark.parametrize("question", ["亚托莉生日是几月几号？", "你的电压是多少？", "你身高多少厘米？", "你的充电接口是啥？", "她充满电可以运行多久？", "她是不是用专用充电器？", "亚托莉需要何种充电器？"])
def test_nearby_but_nonanswering_fact_does_not_authorize_a_spec_guess(question):
    class RAG:
        def search(self, *args):
            return [{"metadata": {"kind": "canonical_fact", "answer": "亚托莉是仿生机器人。", "answer_contract": "{}"}}]
    tasks = make_plan([{"role": "user", "content": question}], RAG())
    assert len(tasks) == 1 and tasks[0].kind == "unknown"


def test_memory_ownership_split_questions_and_no_mutation():
    records = [{"key": "偏好饮品", "value": "黑咖啡"}, {"key": "校验代号", "value": "海盐83"}]
    class RAG:
        def search(self, question, history=()):
            card = next(c for c in CARDS if c["id"] == "identity")
            return [{"metadata": {"kind": "canonical_fact", "answer": card["answer"],
                                   "answer_contract": json.dumps(card["answer_contract"])}}]
    messages = [{"role": "user", "content": "你是机器人吗？另外，我喜欢喝什么？"}]
    original = json.dumps((records, messages))
    tasks = make_plan(messages, RAG(), records)
    answer, trace = render_plan(tasks, "ATRI人设", lambda msgs: "我不是人类，是仿生机器人。")
    assert "机器人" in answer and "主人，你保存的偏好饮品" in answer and "黑咖啡" in answer
    assert "我的最爱" not in answer
    assert [t["kind"] for t in trace] == ["fact", "memory"]
    assert json.dumps((records, messages)) == original


def test_current_explicit_correction_does_not_rewrite_saved_memory():
    records = [{"key": "偏好饮品", "value": "黑咖啡"}]
    text = memory_reference("我喜欢喝什么", records, [{"role": "user", "content": "我现在喜欢喝红茶。"}])
    assert "红茶" in text and "尚未改写" in text
    assert records[0]["value"] == "黑咖啡"


def test_memory_confirmation_of_value_keeps_user_as_subject():
    records = [{"key": "偏好饮品", "value": "黑咖啡"}]
    tasks = make_plan([{"role": "user", "content": "我是不是喜欢黑咖啡？"}], None, records)
    answer, trace = render_plan(tasks, "ATRI", lambda msgs: pytest.fail("do not ask model to assign ownership"))
    assert "主人，你保存的偏好饮品" in answer and "黑咖啡" in answer


@pytest.mark.parametrize("statement,question,value", [
    ("我喜欢喝红茶。", "我喜欢喝什么", "红茶"),
    ("我叫小王。", "我叫什么名字", "小王"),
])
def test_explicit_current_chat_facts_work_without_persisting(statement, question, value):
    records = []
    answer = memory_reference(question, records, [{"role": "user", "content": statement}])
    assert value in answer and "本轮" in answer and records == []


def test_current_chat_is_not_misreported_as_saved_memory():
    history = [{"role": "user", "content": "我喜欢喝红茶。"}]
    assert "没有找到" in memory_reference("我保存的饮品是什么", [], history)
    answer = memory_reference("我保存的饮品是什么", [{"key": "偏好饮品", "value": "黑咖啡"}], history)
    assert "黑咖啡" in answer and "红茶" in answer and "尚未改写" in answer


def test_casual_statements_and_character_names_are_not_split_wrongly():
    assert make_plan([{"role": "user", "content": "我喜欢喝红茶"}], None, []) == []
    assert split_questions("亚托莉和夏生怎么认识的？") == ["亚托莉和夏生怎么认识的"]
    assert make_plan([{"role": "user", "content": "早安亚托莉"}], None, None) == []


def test_too_many_tasks_are_explicitly_declined_not_silently_omitted():
    messages = [{"role": "user", "content": "？".join(["你是谁"] * 20)}]
    tasks = make_plan(messages)
    assert len(tasks) == 1 and tasks[0].kind == "unknown"
    assert "最多六个" in tasks[0].reference


def test_missing_contract_uses_the_reference_not_unchecked_generation():
    task = AnswerTask("资料问题", "fact", "已核实的资料。")
    answer, trace = render_plan([task], "ATRI", lambda msgs: pytest.fail("no unchecked draft"))
    assert "已核实的资料" in answer and trace[0]["method"] == "source_reference"


def test_chat_subtask_keeps_authenticated_users_memory_context(tmp_path):
    from atri_memory import MemoryStore
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.put("alice", "偏好饮品", "乌龙茶")
    store.put("bob", "偏好饮品", "黑咖啡")
    tasks = [AnswerTask("你的充电接口？", "unknown", "不能确认。"),
             AnswerTask("用我喜欢的饮料写一句祝福", "chat")]
    captured = []
    def generate(messages):
        captured.append(json.dumps(messages, ensure_ascii=False))
        return "主人，祝你喝茶时也有好心情。"
    answer, trace = render_plan(tasks, "ATRI", generate,
                                prepare_chat=lambda messages: store.augment("alice", messages))
    assert len(captured) == 1 and "乌龙茶" in captured[0] and "黑咖啡" not in captured[0]
    assert "不能确认" in answer and "好心情" in answer
    assert store.list("alice") == [{"key": "偏好饮品", "value": "乌龙茶"}]


def test_default_persona_template_does_not_leave_unfilled_user_memory(tmp_path, monkeypatch):
    import serve_v1_1_8b as server
    path = tmp_path / "system.txt"
    path.write_text("你是ATRI，称呼{{user}}{{honorific}}。\n记忆：{{memory}}")
    monkeypatch.setattr(server, "SYSTEM_FILE", str(path))
    text = server.default_system()
    assert "{{" not in text and "主人" in text and "按当前用户" in text


@pytest.mark.parametrize("stream", [False, True])
def test_handler_rejects_bad_drafts_before_json_or_sse(monkeypatch, stream):
    import torch
    from fastapi.testclient import TestClient
    import serve_v1_1_8b as server
    card = next(c for c in CARDS if c["id"] == "housework")
    class RAG:
        def search(self, *args):
            return [{"metadata": {"kind": "canonical_fact", "answer": card["answer"],
                                   "answer_contract": json.dumps(card["answer_contract"])}}]
        def augment(self, msgs): pytest.fail("do not mix later scenes into the task")
    monkeypatch.setenv("ATRI_RAG_ENABLED", "1")
    monkeypatch.setenv("ATRI_MEMORY_ENABLED", "0")
    monkeypatch.setenv("ATRI_GROUNDING_ENABLED", "1")
    monkeypatch.setattr("atri_rag.from_env", lambda: RAG())
    class Inputs(dict):
        def to(self, device): return self
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1
        def apply_chat_template(self, *args, **kwargs): return "prompt"
        def __call__(self, *args, **kwargs): return Inputs(input_ids=torch.tensor([[1, 2]]))
        def encode(self, text, **kwargs): return [ord(c) for c in text]
        def decode(self, ids, **kwargs): return "".join(chr(int(c)) for c in ids)
    class Model:
        device = "cpu"
        calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            return torch.tensor([[1, 2] + [ord(c) for c in "她确实很会做饭，已经学会做饭了。"]])
    model = Model()
    seed_events = []
    def seed_inside_generation_lock(value):
        assert server.lock.locked()
        seed_events.append(value)
    monkeypatch.setattr(torch, "manual_seed", seed_inside_generation_lock)
    monkeypatch.setattr(server, "state", {"tok": Tokenizer(), "model": model})
    monkeypatch.setattr(server, "default_system", lambda: "ATRI人设")
    with TestClient(server.build_app()) as client:
        response = client.post("/v1/chat/completions", json={"messages": [
            {"role": "user", "content": "她很会做饭吗？"}], "stream": stream, "seed": 42})
    assert response.status_code == 200
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        answer = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        trace = chunks[-1]["atri"]
    else:
        answer = response.json()["choices"][0]["message"]["content"]
        trace = response.json()["atri"]
    assert card["answer"] in answer
    assert "很会做饭" not in answer and "已经学会做饭" not in answer
    assert model.calls == 2
    assert seed_events == [42, 42]
    assert trace["tasks"][0]["method"] == "source_fallback"


@pytest.mark.parametrize("stream", [False, True])
def test_real_handler_memory_and_unknown_without_generation(tmp_path, monkeypatch, stream):
    import torch
    from fastapi.testclient import TestClient
    import serve_v1_1_8b as server
    from atri_memory import MemoryStore
    token = "private-test-only-token-" * 3
    users = tmp_path / "users.json"
    users.write_text(json.dumps({"alice": token, "bob": "other-test-only-token-" * 3}))
    monkeypatch.setenv("ATRI_MEMORY_ENABLED", "1")
    monkeypatch.setenv("ATRI_MEMORY_DB", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setenv("ATRI_MEMORY_USERS_FILE", str(users))
    monkeypatch.setenv("ATRI_RAG_ENABLED", "1")
    monkeypatch.setenv("ATRI_GROUNDING_ENABLED", "1")
    class RAG:
        def search(self, *args):
            return []
        def augment(self, msgs):
            pytest.fail("planned tasks should not go through the legacy message mixing path")
    monkeypatch.setattr("atri_rag.from_env", lambda: RAG())
    MemoryStore(tmp_path / "memory.sqlite3").put("alice", "偏好饮品", "乌龙茶")
    MemoryStore(tmp_path / "memory.sqlite3").put("bob", "偏好饮品", "黑咖啡")
    class Inputs(dict):
        def to(self, device): return self
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1
        def apply_chat_template(self, *args, **kwargs): return "prompt"
        def __call__(self, *args, **kwargs): return Inputs(input_ids=torch.tensor([[1, 2]]))
        def encode(self, text, **kwargs): return [ord(c) for c in text]
        def decode(self, ids, **kwargs): return "".join(chr(int(c)) for c in ids)
    class Model:
        device = "cpu"
        def generate(self, **kwargs): pytest.fail("direct reference tasks should not invoke generation")
    monkeypatch.setattr(server, "state", {"tok": Tokenizer(), "model": Model()})
    monkeypatch.setattr(server, "default_system", lambda: "ATRI人设")
    with TestClient(server.build_app()) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + token}, json={
            "messages": [{"role": "user", "content": "亚托莉需要什么充电器？另外我喜欢喝什么？"}], "stream": stream})
        assert response.status_code == 200
        if stream:
            chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
            answer = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            answer = response.json()["choices"][0]["message"]["content"]
            assert response.json()["atri"]["generation_attempts"] == 0
        assert "不能确认" in answer and "你保存的" in answer and "乌龙茶" in answer
        assert "黑咖啡" not in answer
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "我喜欢喝什么"}]}).status_code == 401
        assert MemoryStore(tmp_path / "memory.sqlite3").list("alice") == [{"key": "偏好饮品", "value": "乌龙茶"}]
