import json

import pytest

from atri_core.answer_plan import parse_review
from atri_core.memory import LifecycleStore
from atri_core.runtime import Runtime


def hit(ident='school', text='亚托莉想学习知识。', stage='daily', score=.9):
    return {'node_id': ident, 'text': text, 'score': score,
            'metadata': {'kind': 'canonical_summary', 'allowed_stages': [stage],
                'evidence': [{'source': 'fixture.json', 'sha256': 'a' * 64,
                             'scene': 1, 'dialogue_start': 1, 'dialogue_end': 2}],
                'event_time': 'past', 'knowledge_state': stage}}


def review(status='enough', queries=(), required=(), optional=(), gaps=()):
    return dict(status=status, queries=list(queries), must_respond=list(required),
                optional_reference=list(optional), information_gaps=list(gaps))


def run(tmp_path, reviews, search, *, text='为什么去学校？', stage='daily', budget=7000,
        plan=None, store=None):
    store = store or LifecycleStore(tmp_path / 'facts.sqlite3')
    decisions = iter(reviews)
    def generate(messages, *, kind):
        if kind == 'plan':
            return json.dumps(plan or {'utterance_kind': 'other', 'queries': []}, ensure_ascii=False)
        if kind == 'review':
            value = next(decisions)
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return '首次回答保持原样，即使有错误。'
    return Runtime(store, generate, search=search, stage=stage, context_chars=budget,
                   rag_planning=True).reply('fictional', 'event', [{'role': 'user', 'content': text}])


def test_source_bound_handoff_copies_whole_fact_and_boundaries(tmp_path):
    source = hit(text='亚托莉不是只找记忆，她也想学习知识。')
    answer, trace = run(tmp_path, [review(required=['school'])], lambda *args: [source])
    plan = trace['answer_plan']
    assert plan['must_respond'] == [{'source_id': 'school', 'fact': source['text'],
                                    'evidence': source['metadata']['evidence']}]
    assert plan['fact_boundaries'][0]['knowledge_state'] == 'daily'
    assert trace['answer_context']['references'] == []
    assert answer == trace['calls'][-1]['raw_first'] == trace['final']
    assert trace['memory_before'] == trace['memory_after'] == []


def test_two_extra_rounds_excluding_initial_lookup(tmp_path):
    queries = []
    def search(q, history):
        queries.append(q)
        return [hit(q)]
    decisions = [review('lookup', ['补一']), review('lookup', ['补二']),
                 review('lookup', ['不得执行的第三轮'], ['补二'], gaps=['仍缺接口型号'])]
    _, trace = run(tmp_path, decisions, search)
    assert queries == ['为什么去学校？', '补一', '补二']
    assert trace['rag_extra_rounds'] == 2
    assert trace['answer_plan']['stop_reason'] == 'round_limit'
    assert '仍缺接口型号' in trace['answer_plan']['information_gaps']
    assert trace['rag_reviews'][-1]['input']['can_search'] is False


@pytest.mark.parametrize('extra_result', [[], [hit()]])
def test_no_new_admitted_result_stops_and_retains_unknown(tmp_path, extra_result):
    seen = []
    def search(q, history):
        seen.append(q)
        return [hit()] if len(seen) == 1 else extra_result
    _, trace = run(tmp_path, [review('lookup', ['未知接口']),
        review('lookup', ['不该再查'], gaps=['准确接口型号未知'])], search)
    assert len(seen) == 2
    assert trace['answer_plan']['stop_reason'] == 'no_new_results'
    assert trace['answer_plan']['information_gaps']


def test_repeated_normalized_query_does_not_execute(tmp_path):
    calls = []
    _, trace = run(tmp_path, [review('lookup', ['  abc  '])],
        lambda q, h: calls.append(q) or [], text='ABC')
    assert calls == ['ABC'] and trace['rag_extra_rounds'] == 0
    assert trace['answer_plan']['stop_reason'] == 'repeated_query'


@pytest.mark.parametrize('initial_failure', [False, True])
def test_search_failure_is_unknown_and_never_retried(tmp_path, initial_failure):
    seen = []
    def search(q, history):
        seen.append(q)
        if initial_failure or len(seen) > 1:
            raise OSError('fixture failure')
        return []
    decisions = ([review('lookup', ['补查'])] if not initial_failure else []) + [review(gaps=['没有查明'])]
    _, trace = run(tmp_path, decisions, search)
    assert len(seen) == (1 if initial_failure else 2)
    assert trace['answer_plan']['stop_reason'] == 'retrieval_failed'
    assert trace['errors'] == [{'phase': 'rag_lookup', 'error': 'OSError'}]


def test_missing_retriever_cannot_be_invented_by_model(tmp_path):
    _, trace = run(tmp_path, [review('lookup', ['查接口'])], None)
    assert trace['answer_plan']['stop_reason'] == 'retrieval_unavailable'
    assert trace['rag_extra_rounds'] == 0


@pytest.mark.parametrize('bad', ['not json', json.dumps(review(required=['made-up'])),
    json.dumps({**review(), 'stage': 'past_recovered'}), json.dumps(review(queries=['x']))])
def test_malformed_or_fabricated_arrangement_fails_closed(tmp_path, bad):
    _, trace = run(tmp_path, [bad], lambda *args: [hit()])
    assert trace['answer_plan']['must_respond'] == []
    assert trace['answer_plan']['stop_reason'] == 'invalid_review'
    assert trace['answer_plan']['information_gaps']


def test_later_stage_and_unverified_chunks_never_reach_reviewer(tmp_path):
    future = hit('future', '已找回过去记忆', stage='past_recovered')
    unknown = hit('unknown'); unknown['metadata']['evidence'] = []
    _, trace = run(tmp_path, [review('lookup', ['回忆']), review()],
                   lambda *args: [future, unknown])
    assert all(r['input']['references'] == [] for r in trace['rag_reviews'])
    assert trace['answer_context']['references'] == []
    assert trace['retrieval_steps'][0]['returned'] == [future, unknown]
    assert trace['retrieval_steps'][0]['accepted_ids'] == []


def test_context_budget_is_shared_by_sources_and_arrangement(tmp_path):
    sources = [hit(str(i), '完整的事实主体和否定都不截断。' * 12, score=1-i/10) for i in range(5)]
    # Reviewer cannot select a source omitted by the input budget.
    _, trace = run(tmp_path, [review(required=['0'])], lambda *args: sources, budget=2200)
    assert len(json.dumps(trace['answer_context'], ensure_ascii=False)) <= 2200
    assert len(trace['rag_reviews'][0]['input']['references']) < 5
    assert trace['answer_plan']['must_respond'][0]['fact'] == sources[0]['text']


def test_lookup_never_reexecutes_memory_write_or_accepts_operation_proposal(tmp_path):
    store = LifecycleStore(tmp_path / 'operations.sqlite3')
    original = store.observe
    writes = []
    def observe(*args):
        writes.append(args)
        return original(*args)
    store.observe = observe
    _, trace = run(tmp_path, [review('lookup', ['补查']), review()],
        lambda q, h: [hit(q)], store=store, text='我喜欢画画', plan={'utterance_kind': 'self_report'})
    assert len(writes) == 1 and len(store.records('fictional')) == 1
    assert trace['memory_before'] == [] and len(trace['memory_after']) == 1
    assert trace['answer_plan']['operation_results'][0]['status'] == 'saved'
    run(tmp_path, [review()], lambda *args: [], store=store, text='我喜欢画画',
        plan={'utterance_kind': 'self_report'})
    assert len(store.records('fictional')) == 1  # same authenticated event


def test_failed_save_is_copied_from_actual_execution(tmp_path):
    store = LifecycleStore(tmp_path / 'fail.sqlite3')
    with store.connect() as db:
        db.execute("CREATE TRIGGER fail BEFORE INSERT ON facts BEGIN SELECT RAISE(ABORT,'fixture'); END")
    _, trace = run(tmp_path, [review()], lambda *args: [], store=store,
                   text='我喜欢画画', plan={'utterance_kind': 'self_report'})
    assert trace['answer_plan']['operation_results'][0]['status'] == 'failed'
    assert trace['memory_after'] == []


@pytest.mark.parametrize('stream', [False, True])
def test_http_returns_only_first_role_answer_and_server_controls_feature(tmp_path, stream):
    from fastapi.testclient import TestClient
    from atri_core.http import build_app
    def factory(body):
        return lambda messages, kind: (json.dumps({'utterance_kind': 'other'}) if kind == 'plan'
            else json.dumps(review('lookup', ['内部查询'])) if kind == 'review'
            else '角色首答')
    token = 'fixture-token-for-isolated-http-test-only'
    app = build_app(LifecycleStore(tmp_path / 'http.sqlite3'), {'fictional': token}, factory,
        system='角色', identity={'model': 'fixture'}, search=lambda *args: [], rag_planning=True)
    response = TestClient(app).post('/v1/chat/completions', headers={
        'Authorization': 'Bearer ' + token, 'Idempotency-Key': 'event'},
        json={'messages': [{'role': 'user', 'content': '接口型号？'}], 'stream': stream,
              'rag_planning': False, 'stage': 'past_recovered'})
    assert response.status_code == 200
    assert '内部查询' not in response.text
    if not stream:
        assert response.json()['choices'][0]['message']['content'] == '角色首答'
    else:
        chunks = [json.loads(s[6:]) for s in response.text.splitlines() if s.startswith('data: {')]
        assert ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks) == '角色首答'
