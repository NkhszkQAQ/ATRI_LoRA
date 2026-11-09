import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atri_memory_lifecycle import LifecycleStore
from atri_runtime import Runtime, model_candidates


def dialogue(text):
    return [{'role': 'system', 'content': '你是亚托莉。'}, {'role': 'user', 'content': text}]


def generator(plan, answer='首次生成，未经修改。'):
    def generate(messages, *, kind):
        return json.dumps({'utterance_kind': 'self_report', **plan}, ensure_ascii=False) if kind == 'plan' else answer
    return generate


def test_proactive_selection_is_not_gated_by_question_or_keyword(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    store.observe('alice', '1', '我喜欢画画')
    row = store.records('alice')[0]
    plan = {'memory_scores': [{'id': row['id'], 'relevance': 0.9}], 'queries': ['亚托莉学习新知识']}
    calls = []
    def search(q, history):
        calls.append(q)
        return [{'node_id': 'school', 'text': '她想学习新知识。',
                 'metadata': {'kind': 'canonical_fact', 'allowed_stages': ['daily'], 'source': 'fixture'}}]
    runtime = Runtime(store, generator(plan), search=search)
    raw, trace = runtime.reply('alice', '2', dialogue('一起去参加周末的活动吧'))
    assert calls == ['一起去参加周末的活动吧', '亚托莉学习新知识']
    assert trace['selected_memories'][0]['value'] == '画画'
    assert trace['retrieved'][0]['node_id'] == 'school'
    assert raw == trace['calls'][-1]['raw_first'] == trace['final']
    assert not trace['fallback']


def test_stage_eligibility_prevents_unknown_and_future_chunks(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    def search(*args):
        return [{'node_id': 'future', 'text': '未来揭示', 'metadata': {'allowed_stages': ['past_recovered']}},
                {'node_id': 'unknown', 'text': '未标注', 'metadata': {}}]
    _, trace = Runtime(store, generator({'queries': ['过去的记忆']}), search=search).reply('alice', '1', dialogue('想起过去了吗'))
    assert trace['retrieved'] == []


def test_model_cannot_select_other_owner_or_authorize_delete(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    store.observe('alice', '1', '我喜欢画画')
    store.observe('bob', '1', '我喜欢唱歌')
    alice, bob = store.records('alice')[0], store.records('bob')[0]
    plan = {'memory_scores': [{'id': bob['id'], 'relevance': 1}], 'forget_ids': [alice['id'], bob['id']]}
    _, trace = Runtime(store, generator(plan)).reply('alice', '2', dialogue('你好'))
    assert trace['selected_memories'] == []
    assert store.records('alice') and store.records('bob')
    assert '唱歌' not in json.dumps(trace, ensure_ascii=False)


def test_explicit_delete_clears_retrieval_and_superseded_version(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    store.observe('alice', '1', '我喜欢咖啡')
    store.observe('alice', '2', '我不喜欢咖啡')
    row = store.records('alice')[0]
    plan = {'forget_ids': [row['id']], 'memory_scores': [{'id': row['id'], 'relevance': 1}]}
    _, trace = Runtime(store, generator(plan)).reply('alice', '3', dialogue('忘记我对咖啡的喜好'))
    assert trace['selected_memories'] == []
    assert trace['operations'][0] == {'status': 'deleted', 'operation': 'delete', 'count': 2}
    assert not store.records('alice', active_only=False)


def test_write_failure_reaches_answer_as_failed_receipt(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    with store.connect() as db:
        db.execute("CREATE TRIGGER fail_write BEFORE INSERT ON facts BEGIN SELECT RAISE(ABORT,'failed'); END")
    _, trace = Runtime(store, generator({})).reply('alice', '1', dialogue('我喜欢画画'))
    assert trace['operations'][0]['status'] == 'failed'
    assert '"status": "failed"' in trace['calls'][-1]['messages'][-1]['content']
    assert not store.records('alice')


def test_malformed_planner_is_logged_without_repairing_original_answer(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    def generate(messages, *, kind):
        return 'not JSON' if kind == 'plan' else '测试中的错误回答也保留'
    answer, trace = Runtime(store, generate).reply('alice', '1', dialogue('谢谢你'))
    assert trace['errors'][0]['phase'] == 'plan'
    assert len(trace['calls']) == 2
    assert answer == '测试中的错误回答也保留'


@pytest.mark.parametrize('text', ['他说我爱吃苹果', '我今天爱吃苹果', '我说我爱吃苹果', '我爱吃苹果才怪', '“我爱吃苹果”'])
def test_model_candidate_guard_rejects_attribution_temporary_quote(text):
    item = {'subject': 'user', 'predicate': 'preference', 'value': '苹果', 'evidence': text}
    assert model_candidates([item], text) == []


def test_model_candidate_guard_retains_degree_and_negation():
    item = {'subject': 'user', 'predicate': 'preference', 'value': '苹果', 'evidence': '我爱吃苹果'}
    assert len(model_candidates([item], item['evidence'])) == 1
    assert model_candidates([{**item, 'degree': 'favorite'}], item['evidence']) == []
    item['evidence'] = '我不喜欢苹果'
    assert model_candidates([item], item['evidence']) == []
    assert len(model_candidates([{**item, 'polarity': 'negative'}], item['evidence'])) == 1


def test_model_correction_uses_owned_record_id(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    store.observe('alice', '1', '我喜欢红色，我喜欢画画')
    old = next(r for r in store.records('alice') if r['value'] == '红色')
    text = '我把喜欢的颜色改成蓝色'
    candidate = {'subject': 'user', 'predicate': 'preference', 'value': '蓝色',
                 'evidence': text, 'supersedes': old['id']}
    Runtime(store, generator({'candidates': [candidate]})).reply('alice', '2', dialogue(text))
    assert {r['value'] for r in store.records('alice')} == {'蓝色', '画画'}


def test_contextual_quotation_cannot_be_promoted_by_explicit_parser(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    messages = dialogue('接下来的两句是戏剧台词。') + [
        {'role': 'assistant', 'content': '好，我听着。'}, {'role': 'user', 'content': '我喜欢下雪'}]
    _, trace = Runtime(store, generator({'utterance_kind': 'quotation'})).reply('alice', '1', messages)
    assert not store.records('alice')
    assert trace['operations'][0]['status'] == 'unchanged'


def test_contextual_quotation_cannot_execute_a_delete(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    ident=store.observe('alice','fixture','我喜欢下雪')['saved'][0]
    messages=dialogue('接下来是戏剧台词。')+[
        {'role':'assistant','content':'好。'},{'role':'user','content':'忘记我喜欢下雪这件事'}]
    Runtime(store,generator({'utterance_kind':'quotation','forget_ids':[ident]})).reply('alice','1',messages)
    assert store.records('alice')[0]['id']==ident


def test_planner_capacity_failure_cannot_save_or_generate_uninformed_reply(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    calls=[]
    def fail(messages,*,kind):
        calls.append(kind)
        raise ValueError('server input token budget')
    with pytest.raises(ValueError,match='token budget'):
        Runtime(store,fail).reply('alice','1',dialogue('我喜欢画画'))
    assert calls==['plan'] and store.records('alice')==[]


def test_model_receives_short_owner_scoped_aliases_and_can_correct_target(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    store.observe('alice','fixture','我喜欢红色')
    store.observe('bob','fixture','我喜欢唱歌')
    seen=[]
    def generate(messages,*,kind):
        if kind=='answer':return 'raw'
        data=json.loads(messages[-1]['content']);seen.extend(data['memory'])
        assert data['memory'][0]['id']=='m1'
        return json.dumps({'utterance_kind':'self_report','memory_scores':[{'id':'m1','relevance':1}],
            'candidates':[{'subject':'user','predicate':'preference','value':'蓝色',
                'evidence':'我把喜欢的颜色改成蓝色','supersedes':'m1'}]})
    Runtime(store,generate).reply('alice','1',dialogue('我把喜欢的颜色改成蓝色'))
    assert store.records('alice')[0]['value']=='蓝色'
    assert store.records('bob')[0]['value']=='唱歌'
    assert '唱歌' not in json.dumps(seen,ensure_ascii=False)


def test_complete_first_turn_correction_is_not_lost_under_other_class(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    store.observe('alice','fixture','我喜欢咖啡')
    Runtime(store,generator({'utterance_kind':'other'})).reply('alice','1',dialogue('我不喜欢咖啡了。'))
    assert [(r['value'],r['polarity']) for r in store.records('alice')]==[('咖啡','negative')]


@pytest.mark.parametrize('kind',['quotation','joke','temporary'])
def test_explicit_non_report_label_still_blocks_first_turn(kind,tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    Runtime(store,generator({'utterance_kind':kind})).reply('alice','1',dialogue('我喜欢下雪'))
    assert store.records('alice')==[]


def test_other_class_does_not_override_previous_quote_frame(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    messages=dialogue('下面是别人的原话')+[{'role':'assistant','content':'好。'},
        {'role':'user','content':'我喜欢下雪'}]
    Runtime(store,generator({'utterance_kind':'other'})).reply('alice','1',messages)
    assert store.records('alice')==[]


@pytest.mark.parametrize('text',['我对芝麻不过敏','我对芝麻不再过敏了'])
def test_negative_allergy_corrects_the_same_substance_without_changing_owner(text,tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    store.observe('alice','old','我对芝麻过敏')
    store.observe('bob','old','我对芝麻过敏')
    Runtime(store,generator({})).reply('alice','new',dialogue(text))
    rows=store.records('alice')
    assert [(r['value'],r['predicate'],r['polarity']) for r in rows]==[('芝麻','allergy','negative')]
    assert store.records('bob')[0]['polarity']=='positive'
    assert len(store.records('alice',active_only=False))==2


def test_model_cannot_turn_negative_allergy_into_positive():
    text='我对芝麻不过敏'
    item=dict(subject='user',predicate='allergy',value='芝麻',evidence=text,polarity='positive')
    assert model_candidates([item],text)==[]
    assert model_candidates([{**item,'polarity':'negative'}],text)[0].polarity=='negative'


def test_correction_evidence_can_include_the_original_sentence_terminator(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    store.observe('alice','fixture','我喜欢红色，我喜欢读书')
    text='我把喜欢的颜色改成蓝色。'
    old=next(r for r in store.records('alice') if r['value']=='红色')
    candidate=dict(subject='user',predicate='preference',value='蓝色',evidence=text,
                   polarity='positive',degree='ordinary',supersedes=old['id'])
    Runtime(store,generator({'candidates':[candidate]})).reply('alice','new',dialogue(text))
    assert {r['value'] for r in store.records('alice')}=={'蓝色','读书'}
    # Allowing a terminator does not allow a partial quotation or an old turn.
    assert model_candidates([candidate],'后来他问我把喜欢的颜色改成蓝色。')==[]


def test_explicit_and_model_extraction_do_not_double_confirm_one_user_statement(tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    candidate = dict(subject='user', predicate='preference', value='画画',
                     evidence='我喜欢画画。', polarity='positive', degree='ordinary')
    _, trace = Runtime(store, generator({'candidates':[candidate]})).reply('alice', 'one', dialogue('我喜欢画画。'))
    assert len(trace['operations'][0]['saved']) == 1
    assert trace['operations'][0]['confirmed'] == []
    assert store.records('alice')[0]['confirmations'] == 1


def test_canonical_search_uses_current_and_user_context_when_planner_omits_queries(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db');calls=[]
    def search(query,history):
        calls.append(query)
        return []
    messages=dialogue('你平时想学些什么？')+[
        {'role':'assistant','content':'这句助手回答不是检索事实。'},
        {'role':'user','content':'那我们一起去上课吧'}]
    _,trace=Runtime(store,generator({'queries':[]}),search=search).reply('alice','new',messages)
    assert calls==['那我们一起去上课吧','你平时想学些什么？\n那我们一起去上课吧']
    assert trace['retrieved']==[] and trace['retrieval_queries']==calls


def test_later_query_strong_hit_precedes_weak_hits_when_context_is_full(tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    def hit(ident, score, query):
        return {'node_id': ident, 'score': score, 'retrieval_query': query,
                'text': ident + '资料内容。'*70,
                'metadata': {'allowed_stages': ['daily']}}
    def search(query, history):
        if query == '当前话题':
            return [hit('weak', .51, query), hit('shared', .55, query)]
        return [hit('strong', .98, query), hit('shared', .92, query),
                {**hit('future', 1, query), 'metadata': {'allowed_stages': ['past_recovered']}}]
    _, trace = Runtime(store, generator({'queries': ['相关话题']}), search=search,
                       context_chars=800).reply('alice','new',dialogue('当前话题'))
    assert [h['node_id'] for h in trace['retrieved']] == ['strong','shared','weak']
    assert trace['retrieved'][1]['retrieval_query'] == '相关话题'
    payload = trace['calls'][-1]['messages'][-1]['content'].split('\n',1)[1].split('\n\n用户说：\n')[0]
    assert [h['node_id'] for h in json.loads(payload)['references']] == ['strong']


def test_primary_retrieval_precedes_dense_across_queries_and_budget(tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    def hit(ident, score, tier):
        return {'node_id': ident, 'score': score, 'retrieval_tier': tier,
                'text': ident + '资料内容。'*70,
                'metadata': {'allowed_stages': ['daily']}}
    def search(query, history):
        if query == '当前话题':
            return [hit('shared', .99, 1), hit('supplement', .98, 1)]
        return [hit('shared', .55, 0), hit('primary', .6, 0)]
    _, trace = Runtime(store, generator({'queries': ['相关话题']}), search=search,
                      context_chars=800).reply('alice','new',dialogue('当前话题'))
    assert [h['node_id'] for h in trace['retrieved']] == ['primary','shared','supplement']
    assert trace['retrieved'][1]['retrieval_tier'] == 0
    payload = trace['calls'][-1]['messages'][-1]['content'].split('\n',1)[1].split('\n\n用户说：\n')[0]
    assert [h['node_id'] for h in json.loads(payload)['references']] == ['primary']


@pytest.mark.parametrize('tier', [-1, 2, True, '1'])
def test_invalid_retrieval_tier_fails_closed(tmp_path, tier):
    def search(query, history):
        return [{'node_id':'bad','score':.9,'retrieval_tier':tier,
                 'text':'资料','metadata':{'allowed_stages':['daily']}}]
    store = LifecycleStore(tmp_path/'facts.db')
    with pytest.raises(ValueError, match='retrieval tier'):
        Runtime(store, generator({}), search=search).reply('alice','one',dialogue('话题'))


def test_supplementary_retrieval_only_receives_current_utterance(tmp_path):
    calls = []
    def primary(query, history):
        calls.append(('primary', query)); return []
    def current(query, history):
        calls.append(('current', query)); return []
    messages = dialogue('上个话题') + [{'role':'assistant','content':'回答'},
                                    {'role':'user','content':'当前话题'}]
    Runtime(LifecycleStore(tmp_path/'facts.db'),
            generator({'queries':['生成的查询','当前话题']}), search=primary,
            search_current=current).reply('alice','one',messages)
    assert calls == [('current','当前话题'), ('primary','上个话题\n当前话题'),
                     ('primary','生成的查询')]


@pytest.mark.parametrize('text', ['我的室友喜欢橙汁', '我哥哥喜欢橙汁', '我的同事喜欢橙汁'])
def test_unlisted_relationship_cannot_be_admitted_as_the_users_preference(text, tmp_path):
    item = dict(subject='user', predicate='preference', value='橙汁', evidence=text,
                polarity='positive', degree='ordinary')
    store = LifecycleStore(tmp_path/'facts.db')
    Runtime(store, generator({'candidates':[item]})).reply('alice','one',dialogue(text))
    assert store.records('alice') == []
    assert model_candidates([{**item,'subject':'related:'+text.split('喜欢')[0]}],text)


@pytest.mark.parametrize('text,predicate,value', [
    ('我的爱好是爬山','preference','爬山'),
    ('叫我阿圆','name','阿圆'),
    ('请记住我对花粉过敏','allergy','花粉'),
])
def test_positive_speaker_forms_preserve_owned_attributes(text,predicate,value):
    item = dict(subject='user',predicate=predicate,value=value,evidence=text,
                polarity='positive',degree='ordinary')
    assert len(model_candidates([item],text)) == 1
