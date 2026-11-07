"""Actual SQLite effects for model-selected, owner-valid but incorrect IDs."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atri_memory_lifecycle import Candidate, LifecycleStore
from atri_runtime import Runtime


def seed(store, subject, value, event):
    return store.observe('fictional', event, value, [Candidate(
        subject, 'statement', value, value, importance=.8, stability=.95)])['saved'][0]


def run(store, text, ids, event='delete'):
    def generate(messages, *, kind):
        return json.dumps({'utterance_kind': 'other', 'forget_ids': ids}) if kind == 'plan' else 'raw'
    return Runtime(store, generate).reply('fictional', event, [{'role': 'user', 'content': text}])[1]


def test_missing_birthday_cannot_delete_saved_occupation(tmp_path):
    store = LifecycleStore(tmp_path / 'memory.db')
    occupation = seed(store, 'user', '我的职业是木匠', 'job')
    trace = run(store, '请忘记我的生日。', [occupation])
    assert [r['id'] for r in store.records('fictional')] == [occupation]
    assert trace['operations'] == [{'status': 'unavailable', 'operation': 'delete', 'count': 0}]
    assert trace['deletion_validation']['rejected'] == [occupation]


def test_subject_and_attribute_limit_delete_even_when_model_selects_all(tmp_path):
    store = LifecycleStore(tmp_path / 'memory.db')
    birthday = seed(store, 'user', '我的生日是六月六日', 'birthday')
    job = seed(store, 'user', '我的职业是木匠', 'job')
    sister = seed(store, 'related:我姐姐', '我姐姐的生日是六月六日', 'sister')
    trace = run(store, '删除我的出生日期的信息。', [birthday, job, sister])
    assert {r['id'] for r in store.records('fictional')} == {job, sister}
    assert trace['operations'][0]['count'] == 1


@pytest.mark.parametrize('text', ['忘记这个', '忘记我的生日，但保留六月六日',
    '忘记我姐姐的生日', '忘记我的职业', '忘记我的生日和职业', '忘记我的生日是九月九日'])
def test_unresolved_or_mismatched_delete_leaves_record(tmp_path, text):
    store = LifecycleStore(tmp_path / 'memory.db')
    ident = seed(store, 'user', '我的生日是六月六日', 'birthday')
    run(store, text, [ident])
    assert store.records('fictional')[0]['id'] == ident


def test_full_related_assertion_can_be_deleted(tmp_path):
    store = LifecycleStore(tmp_path / 'memory.db')
    ident = seed(store, 'related:我姐姐', '我姐姐是工程师', 'sister')
    trace = run(store, '忘记我姐姐是工程师这件事', [ident])
    assert not store.records('fictional') and trace['operations'][0]['count'] == 1


def test_same_preference_value_does_not_authorize_other_subject(tmp_path):
    store = LifecycleStore(tmp_path / 'memory.db')
    first = store.observe('fictional', 'one', '我喜欢咖啡')['saved'][0]
    other = store.observe('fictional', 'two', '我姐姐喜欢咖啡', [Candidate(
        'related:我姐姐', 'preference', '咖啡', '我姐姐喜欢咖啡')])['saved'][0]
    run(store, '忘记我对咖啡的喜好', [first, other])
    assert [r['id'] for r in store.records('fictional')] == [other]


def test_random_saved_ids_stay_in_trace_not_answer_prompt(tmp_path):
    requests = []
    identities = []
    for index in range(2):
        store = LifecycleStore(tmp_path / f'{index}.db')
        def generate(messages, *, kind):
            if kind == 'plan':
                return json.dumps({'utterance_kind': 'self_report'})
            requests.append(messages)
            return 'raw'
        _, trace = Runtime(store, generate).reply('fictional', 'same-event', [{'role': 'user', 'content': '我喜欢画画'}])
        identities.append(trace['operations'][0]['saved'][0])
        assert identities[-1] not in json.dumps(requests[-1])
        assert '"saved_count": 1' in requests[-1][-1]['content']
    assert identities[0] != identities[1] and requests[0] == requests[1]
