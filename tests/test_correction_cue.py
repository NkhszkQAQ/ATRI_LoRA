"""Accept verbatim correction evidence without erasing attribution or clauses."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atri_memory_lifecycle import Candidate, LifecycleStore
from atri_runtime import Runtime, model_candidates


def item(evidence, value='我的学校是秋野学院', **extra):
    return dict(subject='user', predicate='statement', value=value, evidence=evidence,
                polarity='positive', degree='ordinary', importance=.7, stability=.9, **extra)


@pytest.mark.parametrize('cue', ['更正一下，', '更正, ', '纠正一下，', '纠正，'])
def test_complete_correction_keeps_original_evidence_and_replaces_only_same_attribute(cue, tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    old = '我的学校是青叶学院'
    old_id = store.observe('fictional', 'old', old, [Candidate('user', 'statement', old, old)])['saved'][0]
    language = '我的母语是日语'
    store.observe('fictional', 'language', language, [Candidate('user', 'statement', language, language)])
    current = cue + '我的学校是秋野学院。'
    def generate(messages, *, kind):
        if kind == 'answer':
            return '[fixture answer; not an 8B acceptance result]'
        inventory = json.loads(messages[-1]['content'])['memory']
        alias = next(m['id'] for m in inventory if m['value'] == old)
        return json.dumps({'utterance_kind': 'self_report', 'candidates': [item(current, supersedes=alias)]})
    _, trace = Runtime(store, generate).reply('fictional', 'new', [{'role': 'user', 'content': current}])
    assert trace['operations'][0]['status'] == 'saved'
    rows = LifecycleStore(tmp_path / 'facts.db').records('fictional')
    assert {r['value'] for r in rows} == {language, '我的学校是秋野学院'}
    corrected = next(r for r in rows if r['value'] != language)
    assert corrected['evidence'] == current and corrected['supersedes'] == old_id
    assert not store.records('different-fictional-owner')


@pytest.mark.parametrize('current,evidence,value', [
    ('她说，更正一下，我的学校是秋野学院。', '更正一下，我的学校是秋野学院。', '我的学校是秋野学院'),
    ('更正一下，我的学校是秋野学院，但这是假设。', '更正一下，我的学校是秋野学院', '我的学校是秋野学院'),
    ('更正一下，我的学校是秋野学院，这是台词。', '更正一下，我的学校是秋野学院', '我的学校是秋野学院'),
    ('更正一下，我的学校是秋野学院。只是玩笑。', '更正一下，我的学校是秋野学院。', '我的学校是秋野学院'),
    ('“更正一下，我的学校是秋野学院。”', '更正一下，我的学校是秋野学院。', '我的学校是秋野学院'),
    ('更正一下，我可能是老师。', '更正一下，我可能是老师。', '我可能是老师'),
    ('更正一下，我明天住在南区。', '更正一下，我明天住在南区。', '我明天住在南区'),
    ('更正一下，我妹妹的学校是秋野学院。', '更正一下，我妹妹的学校是秋野学院。', '我的学校是秋野学院'),
    ('更正一下，我不是老师。', '更正一下，我不是老师。', '我是老师'),
    ('如果更正一下，我的学校是秋野学院。', '如果更正一下，我的学校是秋野学院。', '我的学校是秋野学院'),
])
def test_correction_cue_cannot_remove_context_subject_or_negation(current, evidence, value):
    assert not model_candidates([item(evidence, value)], current)


def test_related_subject_and_negative_statement_remain_verbatim():
    current = '更正一下，我妹妹的学校不是秋野学院。'
    proposed = item(current, '我妹妹的学校不是秋野学院')
    proposed['subject'] = 'related:我妹妹'
    candidates = model_candidates([proposed], current)
    assert len(candidates) == 1
    assert candidates[0].value == '我妹妹的学校不是秋野学院'
    assert candidates[0].evidence == current
