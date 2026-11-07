"""Current corrections withhold conflicting references without pretending to save."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atri_memory_lifecycle import Candidate, LifecycleStore, Policy
from atri_runtime import Runtime


OLD = '我的生日是三月四日'
NEW = '我的生日是七月八日'


def setup(store):
    def save(owner, event, value):
        return store.observe(owner, event, value, [Candidate('user', 'statement', value, value)])['saved'][0]
    birthday = save('fictional', 'birthday', OLD)
    job = save('fictional', 'job', '我的职业是园丁')
    other = save('other-fictional', 'birthday', OLD)
    return birthday, job, other


def run(store, text, *, kind='self_report', candidate=None, event='new', history=None):
    def generate(messages, *, kind):
        if kind == 'answer':
            return '[fixture first answer, not an 8B acceptance result]'
        data = json.loads(messages[-1]['content'])
        scores = [{'id': r['id'], 'relevance': 1} for r in data['memory']]
        return json.dumps({'utterance_kind': frame, 'memory_scores': scores,
                           'candidates': [candidate] if candidate is not None else []})
    frame = kind
    messages = history or [{'role': 'user', 'content': '继续刚才的话题。'},
                           {'role': 'assistant', 'content': '[fixture history]'}]
    answer, trace = Runtime(store, generate).reply('fictional', event,
        [*messages, {'role': 'user', 'content': text}])
    packed = trace['calls'][-1]['messages'][-1]['content'].split('\n', 1)[1].split('\n\n用户说：\n')[0]
    return answer, trace, json.loads(packed)


def test_real_missing_evidence_plan_cannot_reintroduce_the_conflicting_old_fact(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    birthday, job, _ = setup(store)
    before = store.records('fictional', active_only=False)
    missing = {'subject': 'user', 'predicate': 'statement', 'value': NEW,
               'polarity': 'positive', 'degree': 'ordinary', 'supersedes': 'm1'}
    answer, trace, context = run(store, '更正一下，' + NEW + '。', candidate=missing)
    assert store.records('fictional', active_only=False) == before
    assert [r['id'] for r in trace['selected_memories']] == [job]
    assert trace['unresolved_corrections'][0]['old_id'] == birthday
    assert context['unresolved_corrections'] == [{'subject': 'user', 'value': NEW,
        'evidence': '更正一下，' + NEW + '。', 'new_value_persisted': False}]
    assert context['operations'][-1]['status'] == 'unavailable'
    assert context['operations'][-1]['operation'] == 'correct'
    assert birthday not in json.dumps(context)
    assert json.loads(trace['calls'][0]['raw_first'])['candidates'] == [missing]
    assert answer == trace['calls'][-1]['raw_first'] == trace['final']
    assert store.records('other-fictional')[0]['value'] == OLD


def test_exclusion_precedes_budget_and_does_not_touch_storage(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db', Policy(limit=1, minimum_relevance=0))
    birthday, job, other = setup(store)
    before = store.records('fictional', active_only=False)
    observed = []
    def scores(rows):
        observed.extend(r['id'] for r in rows)
        return [1 for r in rows]
    assert [r['id'] for r in store.rank('fictional', scores, excluded_ids=[birthday, other])] == [job]
    assert observed == [job]
    assert store.records('fictional', active_only=False) == before
    assert len(store.records('other-fictional')) == 1


@pytest.mark.parametrize('frame', ['quotation', 'joke', 'temporary', 'other'])
def test_non_assertion_frame_does_not_claim_an_unapplied_correction(frame, tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    birthday, job, _ = setup(store)
    _, trace, context = run(store, '更正一下，' + NEW + '。', kind=frame)
    assert 'unresolved_corrections' not in trace and 'unresolved_corrections' not in context
    assert {r['id'] for r in trace['selected_memories']} == {birthday, job}


@pytest.mark.parametrize('text', [
    '“更正一下，我的生日是七月八日。”',
    '如果更正一下，我的生日是七月八日。',
    '不要更正，我的生日是七月八日。',
    '更正一下，我的生日可能是七月八日。',
    '更正一下，我的生日是七月八日，开玩笑的。',
    '她说，更正一下，我的生日是七月八日。',
    '更正一下，我姐姐的生日是七月八日。',
    '更正一下，我的学校是青叶学院。',
    '我的生日是三月四日。',
    '我的生日是七月八日。',
    '更正一下，我的生日是七月八日吗？',
])
def test_unsupported_or_ambiguous_scope_leaves_references_unchanged(text, tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    birthday, job, _ = setup(store)
    _, trace, context = run(store, text)
    assert 'unresolved_corrections' not in context
    assert {r['id'] for r in trace['selected_memories']} == {birthday, job}


def test_multiple_old_attribute_records_do_not_guess_a_target(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    setup(store)
    another = '我的生日是六月六日'
    store.observe('fictional', 'ambiguous', another, [Candidate('user', 'statement', another, another)])
    _, trace, context = run(store, '更正一下，' + NEW + '。')
    assert len(trace['selected_memories']) == 3 and 'unresolved_corrections' not in context


def valid_candidate(birthday, text):
    return {'subject': 'user', 'predicate': 'statement', 'value': NEW, 'evidence': text,
            'polarity': 'positive', 'degree': 'ordinary', 'importance': .8, 'stability': 1,
            'supersedes': birthday}


def test_successful_correction_has_no_false_unavailable_receipt(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    birthday, job, _ = setup(store)
    text = '更正一下，' + NEW + '。'
    _, trace, context = run(store, text, candidate=valid_candidate(birthday, text))
    assert {r['value'] for r in store.records('fictional')} == {NEW, '我的职业是园丁'}
    assert trace['operations'][0]['status'] == 'saved'
    assert 'unresolved_corrections' not in context
    assert birthday not in [r['id'] for r in trace['selected_memories']]


def test_failed_transaction_retains_old_storage_but_withholds_conflicting_reference(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    birthday, job, _ = setup(store)
    before = store.records('fictional', active_only=False)
    with store.connect() as db:
        db.execute("CREATE TRIGGER fail_correct BEFORE INSERT ON facts BEGIN SELECT RAISE(ABORT,'fixture failure'); END")
    text = '更正一下，' + NEW + '。'
    _, trace, context = run(store, text, candidate=valid_candidate(birthday, text))
    assert trace['operations'][0]['status'] == 'failed'
    assert context['operations'][-1]['status'] == 'unavailable'
    assert store.records('fictional', active_only=False) == before
    assert [r['id'] for r in trace['selected_memories']] == [job]


def test_duplicate_failed_admission_remains_unresolved_without_confirming_old_fact(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    setup(store)
    text = '更正一下，' + NEW + '。'
    run(store, text)
    before = store.records('fictional', active_only=False)
    _, trace, context = run(store, text)
    assert trace['operations'][0]['status'] == 'duplicate'
    assert context['unresolved_corrections'][0]['new_value_persisted'] is False
    assert store.records('fictional', active_only=False) == before
