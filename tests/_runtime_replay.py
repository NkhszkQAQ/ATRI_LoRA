"""Deterministic fictional replay for comparing pre/post-integration contracts.

This never loads a model. It records exact model messages, operation traces and
SQLite records, so moving implementations cannot silently change the pipeline.
"""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import uuid


def snapshot(tools):
    sys.path.insert(0, str(tools))
    import atri_memory_lifecycle as memory
    from atri_runtime import Runtime
    counter = itertools.count(1)
    memory.uuid.uuid4 = lambda: uuid.UUID(int=next(counter))
    result = []
    with tempfile.TemporaryDirectory() as folder:
        store = memory.LifecycleStore(Path(folder) / "fictional.sqlite3", clock=lambda: 1700000000.)
        def run(owner, event, text, plan, *, history=(), search=None):
            def generate(messages, *, kind):
                return json.dumps(plan, ensure_ascii=False) if kind == "plan" else "未经重写的首次回答。"
            answer, trace = Runtime(store, generate, search=search).reply(owner, event, [
                {"role": "system", "content": "你是亚托莉。"}, *history,
                {"role": "user", "content": text}])
            result.append({"event": event, "answer": answer, "trace": trace,
                           "alice": store.records("fictional:alice", active_only=False),
                           "bob": store.records("fictional:bob", active_only=False)})

        run("fictional:alice", "save", "我喜欢乌龙茶", {"utterance_kind": "self_report"})
        run("fictional:alice", "read", "我喜欢喝什么？", {
            "utterance_kind": "other", "memory_scores": [{"id": "m1", "relevance": .9}]})
        run("fictional:bob", "isolation", "我喜欢喝什么？", {"utterance_kind": "other"})
        run("fictional:alice", "negative", "我不喜欢乌龙茶了", {"utterance_kind": "self_report"})
        run("fictional:alice", "quote", "我是老师", {"utterance_kind": "self_report"}, history=[
            {"role": "user", "content": "下面是戏剧台词"},
            {"role": "assistant", "content": "我在听。"}])
        run("fictional:alice", "malformed", "你好", {})
        run("fictional:alice", "retrieval", "你以前认识谁？", {"utterance_kind": "other"}, search=lambda *a: [
            {"node_id": "later", "text": "后期资料", "score": .99,
             "metadata": {"allowed_stages": ["past_recovered"]}},
            {"node_id": "daily", "text": "虚构的日常资料", "score": .8,
             "metadata": {"allowed_stages": ["daily"]}}])
        run("fictional:alice", "delete", "忘记我的乌龙茶喜好", {
            "utterance_kind": "other", "forget_ids": ["m1"]})
        run("fictional:alice", "delete", "忘记我的乌龙茶喜好", {
            "utterance_kind": "other", "forget_ids": ["m1"]})
    return result


if __name__ == "__main__":
    print(json.dumps(snapshot(Path(sys.argv[1])), ensure_ascii=False, sort_keys=True))
