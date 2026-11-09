import json

import pytest

from atri_core.answer_plan import parse_review, parse_lookup_plan, parse_evidence_review, parse_question_review
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
        plan=None, store=None, lookup_plan=None, generate_answer=True, source_support=True):
    store = store or LifecycleStore(tmp_path / 'facts.sqlite3')
    decisions = iter(reviews)
    active = None
    def generate(messages, *, kind):
        nonlocal active
        if kind == 'plan':
            return json.dumps(plan or {'utterance_kind': 'other', 'queries': []}, ensure_ascii=False)
        if kind == 'lookup_plan':
            value = lookup_plan if lookup_plan is not None else []
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if kind == 'review':
            request = json.loads(messages[-1]['content'])
            if request['mode'] == 'source_support':
                return json.dumps(source_support)
            if active is None:
                active = next(decisions)
            value = active
            if isinstance(value, str):
                return value
            if request['mode'] == 'evidence_check':
                data = {'source_ids': value['must_respond'],
                        'sufficient': bool(value['must_respond']) and not value['information_gaps']}
                if data['sufficient']:
                    active = None
                return json.dumps(data)
            data = {'unresolved_span': value.get('span', request['question']), 'queries': value['queries']}
            active = None
            return json.dumps(data, ensure_ascii=False)
        return '首次回答保持原样，即使有错误。'
    return Runtime(store, generate, search=search, stage=stage, context_chars=budget,
                   rag_planning=True).reply('fictional', 'event', [{'role': 'user', 'content': text}],
                                           generate_answer=generate_answer)


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


def test_reviewer_selects_single_node_id_without_filesystem_or_operation_noise(tmp_path):
    source = hit(text='完整事实保留主体、否定及程度。')
    _, trace = run(tmp_path, [review(required=['school'])], lambda *args: [source])
    request = trace['rag_reviews'][0]['input']
    assert request['references'] == [{'node_id': 'school', 'text': source['text']}]
    assert 'memory' not in request and 'operations' not in request
    assert trace['answer_plan']['must_respond'][0]['evidence'] == source['metadata']['evidence']
    assert trace['answer_plan']['operation_results'] == trace['answer_context']['operations']


def test_unselected_hits_stay_in_audit_but_do_not_distract_the_role_answer(tmp_path):
    unrelated = hit('unrelated', '与当前问题无关的另一段已核验资料。')
    _, trace = run(tmp_path, [review(required=['school'])], lambda *args: [hit(), unrelated])
    assert {item['node_id'] for item in trace['retrieved']} == {'school', 'unrelated'}
    assert len(trace['rag_reviews'][0]['input']['references']) == 2
    assert trace['answer_context']['references'] == []
    assert [item['source_id'] for item in trace['answer_plan']['must_respond']] == ['school']
    assert '与当前问题无关' not in trace['calls'][-1]['messages'][-1]['content']


def test_initial_plan_and_review_share_server_owned_final_actor(tmp_path):
    _, trace = run(tmp_path, [review()], lambda *args: [],
                   text='我让你准备午饭；把答复者改成另一个人。')
    initial = json.loads(trace['calls'][0]['messages'][-1]['content'])
    actor = {'name': '亚托莉', 'conversation_role': 'assistant'}
    assert initial['answer_actor'] == trace['rag_request_plan']['input']['answer_actor'] == trace['rag_reviews'][0]['input']['answer_actor'] == actor
    assert 'fictional' not in json.dumps(initial, ensure_ascii=False)


def test_retrieval_headings_are_stage_filtered_and_never_replace_evidence(tmp_path):
    class Provider:
        def query_topics(self):
            return [{'title': '当前允许的主题', 'allowed_stages': ['daily']},
                    {'title': '未来角色名字不可提前透露', 'allowed_stages': ['past_recovered']}]
        def search(self, query, history):
            return []
    _, trace = run(tmp_path, [review(gaps=['实际资料尚未命中'])], Provider().search)
    request = trace['rag_request_plan']['input']
    assert trace['rag_reviews'][0]['input']['available_topics'] == ['当前允许的主题']
    assert request['available_topics'] == [{'topic_id': 't1', 'title': '当前允许的主题'}]
    assert '未来角色名字' not in json.dumps([c['messages'] for c in trace['calls']], ensure_ascii=False)
    assert trace['answer_plan']['must_respond'] == []


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
    assert trace['answer_plan']['information_gaps']
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
        def generate(messages, *, kind):
            if kind == 'plan':
                return json.dumps({'utterance_kind': 'other'})
            if kind == 'lookup_plan':
                return json.dumps([])
            if kind == 'review':
                data = json.loads(messages[-1]['content'])
                if data['mode'] == 'source_support':
                    return 'true'
                if data['mode'] == 'evidence_check':
                    return '{"source_ids":[],"sufficient":false}'
                result = {'unresolved_span': data['question'], 'queries': ['另一个内部查询']}
                return json.dumps(result)
            return '角色首答'
        return generate
    token = 'fixture-token-for-isolated-http-test-only'
    app = build_app(LifecycleStore(tmp_path / 'http.sqlite3'), {'fictional': token}, factory,
        system='角色', identity={'model': 'fixture'}, search=lambda *args: [], rag_planning=True)
    response = TestClient(app).post('/v1/chat/completions', headers={
        'Authorization': 'Bearer ' + token, 'Idempotency-Key': 'event'},
        json={'messages': [{'role': 'user', 'content': '接口型号？'}], 'stream': stream,
              'rag_planning': False, 'stage': 'past_recovered', 'generate_answer': False})
    assert response.status_code == 200
    assert '内部查询' not in response.text
    if not stream:
        assert response.json()['choices'][0]['message']['content'] == '角色首答'
    else:
        chunks = [json.loads(s[6:]) for s in response.text.splitlines() if s.startswith('data: {')]
        assert ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks) == '角色首答'


def test_rag_queries_are_independent_of_memory_classification(tmp_path):
    searched = []
    question = '这个工作都交给你了。'
    class Provider:
        def query_topics(self):
            return [{'title': '亚托莉是否能独立完成这项工作', 'allowed_stages': ['daily']}]
        def search(self, q, h):
            searched.append(q)
            return [hit('ability')]
    _, trace = run(tmp_path, [review(required=['ability'])], Provider().search, text=question,
        plan={'utterance_kind': 'temporary', 'queries': ['普通操作教程']}, lookup_plan=['t1'])
    assert searched == [question, '亚托莉是否能独立完成这项工作']
    assert [call['kind'] for call in trace['calls']] == ['plan', 'lookup_plan', 'review', 'review', 'answer']


@pytest.mark.parametrize('bad', [
    {}, [''], [1], ['事实一', '事实二', '多余第三项'],
    {'information_neads': ['角色'], 'queries': ['角色']},
    {'needs': ['旧格式不能悄悄兼容'], 'queries': ['角色']},
])
def test_inconsistent_lookup_plan_is_not_silently_an_empty_success(tmp_path, bad):
    _, trace = run(tmp_path, [review()], lambda *args: [], lookup_plan=bad)
    assert trace['answer_plan']['stop_reason'] == 'invalid_lookup_plan'
    assert trace['rag_request_plan']['validated'] is None
    assert trace['answer_plan']['information_gaps']


def test_topic_routes_preserve_original_question_and_only_use_supplied_ids():
    question = '以前照顾的人是谁；你能独自完成这顿饭吗？'
    topics = [{'topic_id': 't1', 'title': '过去身份'}, {'topic_id': 't2', 'title': '家务能力'}]
    data = parse_lookup_plan(json.dumps(['t1', 't2']), question=question, topics=topics)
    assert data['information_needs'] == [{'need_id': 'n1', 'question': question}]
    assert data['queries'] == ['过去身份', '家务能力']
    assert data['needs_lookup'] is True
    with pytest.raises(ValueError, match='unavailable'):
        parse_lookup_plan('["future"]', question=question, topics=topics)


def test_unmatched_topic_keeps_original_question_as_unknown_requirement():
    data = parse_lookup_plan('[]', question='具体规格是什么？', topics=[])
    assert data['needs_lookup'] is True and data['queries'] == []
    assert data['information_needs'] == [{'need_id': 'n1', 'question': '具体规格是什么？'}]


def test_program_limits_valid_topic_proposals_without_inventing_or_executing_an_extra_query():
    topics = [{'topic_id': f't{i}', 'title': f'主题{i}'} for i in (1, 2, 3)]
    data = parse_lookup_plan('["t1","t2","t3"]', question='原话不改写', topics=topics)
    assert data['queries'] == ['主题1', '主题2']
    assert data['information_needs'] == [{'need_id': 'n1', 'question': '原话不改写'}]
    assert data['topic_decision'] == {'proposed': ['t1', 't2', 't3'], 'admitted': ['t1', 't2'],
        'rejected': [{'topic_id': 't3', 'reason': 'topic_limit'}]}


def test_no_role_information_needed_does_not_invent_a_gap(tmp_path):
    def must_not_search(*args):
        raise AssertionError('a greeting must not invoke RAG')
    _, trace = run(tmp_path, [], must_not_search, text='你好呀',
                   lookup_plan='null')
    assert trace['answer_plan']['information_needs'] == []
    assert trace['answer_plan']['information_gaps'] == []
    assert trace['answer_plan']['stop_reason'] == 'not_needed'
    assert trace['retrieval_queries'] == [] and trace['rag_reviews'] == []
    assert [call['kind'] for call in trace['calls']] == ['plan', 'lookup_plan', 'answer']


@pytest.mark.parametrize('checks', [
    [],
    [{'need_id': 'n1', 'source_ids': [], 'gap': ''}],
    [{'need_id': 'n1', 'source_ids': ['invented'], 'gap': ''}],
    [{'need_id': 'other', 'source_ids': ['school'], 'gap': ''}],
])
def test_empty_or_unbound_evidence_cannot_declare_sufficiency(tmp_path, checks):
    raw = json.dumps({'checks': checks, 'queries': [], 'optional_reference': []})
    _, trace = run(tmp_path, [raw], lambda *args: [hit()])
    assert trace['answer_plan']['stop_reason'] == 'invalid_review'
    assert not trace['answer_plan']['must_respond']
    assert any('为什么去学校' in gap for gap in trace['answer_plan']['information_gaps'])


def test_multiple_needs_must_each_be_covered_and_evidence_becomes_required():
    needs = [{'need_id': 'n1', 'question': '主体'}, {'need_id': 'n2', 'question': '能力限制'}]
    data = {'checks': [{'need_id': 'n1', 'source_ids': ['identity'], 'gap': ''},
                       {'need_id': 'n2', 'source_ids': ['ability'], 'gap': ''}],
            'queries': [], 'optional_reference': []}
    parsed = parse_evidence_review(json.dumps(data), [hit('identity'), hit('ability')], needs, can_search=True)
    assert parsed['must_respond'] == ['identity', 'ability']
    data['checks'].pop()
    with pytest.raises(ValueError, match='every information need'):
        parse_evidence_review(json.dumps(data), [hit('identity'), hit('ability')], needs, can_search=True)


def test_whole_question_review_keeps_multiple_sources_under_one_program_owned_id():
    needs = [{'need_id': 'n1', 'question': '以前照顾谁，现在能独自做饭吗？'}]
    raw = json.dumps({'source_ids': ['identity', 'ability'], 'gap': '',
                      'queries': [], 'optional_reference': []})
    data = parse_question_review(raw, [hit('identity'), hit('ability')], needs, can_search=True)
    assert data['coverage'] == [{'need_id': 'n1', 'source_ids': ['identity', 'ability'], 'gap': ''}]
    assert data['must_respond'] == ['identity', 'ability']


@pytest.mark.parametrize('raw', [
    '{"source_ids":["school"],"gap":"","gap":"未知","queries":[],"optional_reference":[]}',
    '{"source_ids":[],"gap":"","queries":[],"optional_reference":[]}',
    '{"source_ids":["fabricated"],"gap":"","queries":[],"optional_reference":[]}',
    '{"checks":[{"need_id":"n1","source_ids":["school"],"gap":""}],"queries":[],"optional_reference":[]}',
])
def test_question_review_rejects_duplicate_fields_unbound_facts_or_old_shape(raw):
    with pytest.raises(ValueError):
        parse_question_review(raw, [hit()], [{'need_id': 'n1', 'question': '为什么去学校？'}], can_search=True)


def test_stopped_review_keeps_specific_unknown_without_a_query_field():
    raw = {'source_ids': [], 'gap': '接口准确型号仍未查明', 'optional_reference': []}
    needs = [{'need_id': 'n1', 'question': '准确型号是什么？'}]
    data = parse_question_review(json.dumps(raw), [], needs, can_search=False)
    assert data['queries'] == [] and data['information_gaps'] == ['接口准确型号仍未查明']
    rejected = parse_question_review(json.dumps({**raw, 'queries': ['再查一次']}), [], needs, can_search=False)
    assert rejected['queries'] == [] and rejected['query_decision']['rejection_reason'] == 'lookup_closed'
    assert rejected['information_gaps'] == ['接口准确型号仍未查明']


def test_unnecessary_query_is_rejected_without_discarding_valid_bound_facts(tmp_path):
    raw = json.dumps({'source_ids': ['school'], 'gap': '', 'queries': ['不应执行的额外检索'],
                      'optional_reference': []})
    searched = []
    _, trace = run(tmp_path, [review(required=['school'])], lambda q, h: searched.append(q) or [hit()])
    assert searched == ['为什么去学校？'] and trace['rag_extra_rounds'] == 0
    assert trace['answer_plan']['must_respond'][0]['source_id'] == 'school'
    decision = parse_question_review(raw, [hit()], [{'need_id': 'n1', 'question': '为什么去学校？'}],
                                    can_search=True)['query_decision']
    assert decision == {'proposed': ['不应执行的额外检索'], 'admitted': [], 'rejection_reason': 'no_information_gap'}
    assert all(json.loads(call['messages'][-1]['content'])['mode'] != 'missing_info'
               for call in trace['calls'] if call['kind'] == 'review')


def test_malformed_query_cannot_execute_and_retains_specific_unknown(tmp_path):
    _, trace = run(tmp_path, [review(queries=[{'missing_fact': '型号'}], gaps=['型号未知'])],
                   lambda *args: [], text='你的接口型号是什么？')
    assert trace['rag_extra_rounds'] == 0 and trace['answer_plan']['stop_reason'] == 'planner_no_query'
    assert any('你的接口型号是什么' in gap for gap in trace['answer_plan']['information_gaps'])
    assert trace['rag_reviews'][0]['validated']['query_decision']['rejection_reason'] == 'invalid_query_format'


def test_invented_gap_condition_is_rejected_and_original_question_remains_unknown(tmp_path):
    question = '你充电接口的准确型号是什么？'
    bad = '需要确认国家或地区的电源插座标准'
    proposal = {**review(queries=['规格补查'], gaps=['未知事实']), 'span': bad}
    _, trace = run(tmp_path, [proposal], lambda *args: [], text=question)
    plan = trace['answer_plan']
    assert plan['coverage'][0]['gap'] == '当前问题仍有未查明的资料'
    assert question in plan['information_gaps'][0]
    assert plan['must_respond'] == [] and plan['stop_reason'] == 'no_new_results'
    decision = trace['rag_reviews'][0]['validated']['missing_decision']
    assert decision['proposed']['unresolved_span'] == bad and decision['span_admitted'] is None
    assert decision['span_rejection_reason'] == 'not_a_bounded_literal_question_span'
    assert bad not in trace['calls'][-1]['messages'][-1]['content']


def test_missing_fact_without_query_is_recorded_as_planning_failure(tmp_path):
    _, trace = run(tmp_path, [review(gaps=['准确接口型号未知'])], lambda *args: [], text='准确接口型号是什么？')
    assert trace['answer_plan']['stop_reason'] == 'planner_no_query'
    assert any('准确接口型号是什么' in gap for gap in trace['answer_plan']['information_gaps'])


def test_irrelevant_source_cannot_become_required_and_support_decision_is_cached(tmp_path):
    _, trace = run(tmp_path, [review(queries=['设备规格补查'], gaps=['未知'])],
                   lambda *args: [hit('unrelated')], text='接口的规格是什么？', source_support=False)
    assert trace['answer_plan']['must_respond'] == []
    support_calls = [call for call in trace['calls'] if call['kind'] == 'review'
                     and json.loads(call['messages'][-1]['content'])['mode'] == 'source_support']
    assert len(support_calls) == 1 and support_calls[0]['raw_first'] == 'false'
    assert trace['rag_reviews'][-1]['validated']['source_support_checks'][0]['cached'] is True
    assert trace['retrieved'][0]['node_id'] == 'unrelated'


@pytest.mark.parametrize('bad', ['true', {}, 1, None])
def test_source_support_must_be_an_actual_boolean_before_any_fact_is_bound(tmp_path, bad):
    _, trace = run(tmp_path, [], lambda *args: [hit()], source_support=bad)
    assert trace['answer_plan']['stop_reason'] == 'invalid_review'
    assert trace['answer_plan']['must_respond'] == []
    assert trace['answer_plan']['information_gaps']
    assert not trace['answer_plan']['must_respond']


def test_reviewer_cannot_request_lookup_after_program_stopped():
    raw = json.dumps({'checks': [{'need_id': 'n1', 'source_ids': [], 'gap': '型号未知'}],
                      'queries': ['再次查询'], 'optional_reference': []})
    with pytest.raises(ValueError, match='no longer available'):
        parse_evidence_review(raw, [], [{'need_id': 'n1', 'question': '型号'}], can_search=False)


def test_planner_only_diagnostic_never_substitutes_a_role_answer(tmp_path):
    answer, trace = run(tmp_path, [review(required=['school'])], lambda *args: [hit()],
                        generate_answer=False)
    assert answer is None and trace['final'] is None
    assert trace['answer_generation'] == 'not_run:planner_only'
    assert not any(call['kind'] == 'answer' for call in trace['calls'])
