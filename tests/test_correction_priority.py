"""Corrections use explicit current evidence, not an arbitrary score advantage."""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from atri_memory_lifecycle import Candidate,LifecycleStore
from atri_runtime import Runtime


def statement(text,**kwargs):
    return Candidate('user','statement',text.rstrip('。'),text,importance=.8,**kwargs)


def prepare(store):
    birthday=store.observe('alice','birthday','我的生日是三月四日',[statement('我的生日是三月四日')])['saved'][0]
    job=store.observe('alice','job','我的职业是园丁',[statement('我的职业是园丁')])['saved'][0]
    return birthday,job


@pytest.mark.parametrize('stability,importance',[(.65,.8),(.1,0.)])
def test_current_explicit_correction_overrides_importance_and_stability(stability,importance,tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    birthday,job=prepare(store)
    text='更正一下，我的生日是七月八日。'
    c=replace(statement('我的生日是七月八日。',supersedes=birthday,stability=stability),importance=importance)
    receipt=store.observe('alice','correction',text,[c,c])
    assert receipt['status']=='saved' and len(receipt['saved'])==1
    active=store.records('alice')
    updated=next(r for r in active if r['id']!=job)
    assert updated['value']=='我的生日是七月八日' and updated['supersedes']==birthday
    assert updated['stability']==stability and updated['importance']==importance
    assert updated['confirmations']==1
    assert {r['id'] for r in active}=={job,updated['id']}
    assert store.observe('alice','correction',text,[c])['status']=='duplicate'
    reopened=LifecycleStore(tmp_path/'facts.db')
    assert [r['id'] for r in reopened.records('alice')]==[r['id'] for r in active]
    assert reopened.forget_event('alice','forget','忘记生日',[updated['id']])['count']==2
    assert [r['id'] for r in reopened.records('alice',active_only=False)]==[job]


def test_literal_negation_of_an_existing_assertion_has_correction_priority(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    old=store.observe('alice','one','我是园丁',[statement('我是园丁')])['saved'][0]
    receipt=store.observe('alice','two','我不是园丁',[statement('我不是园丁',stability=.2)])
    assert receipt['status']=='saved'
    row=store.records('alice')[0]
    assert row['value']=='我不是园丁' and row['supersedes']==old


@pytest.mark.parametrize('text,evidence,subject,target',[
    ('我的生日是七月八日。','我的生日是七月八日。','user','birthday'),
    ('不要更正，我的生日是七月八日。','我的生日是七月八日。','user','birthday'),
    ('如果要更正，我的生日是七月八日。','我的生日是七月八日。','user','birthday'),
    ('可能要更正一下，我的生日是七月八日。','我的生日是七月八日。','user','birthday'),
    ('更正一下，我的生日可能是七月八日。','我的生日可能是七月八日。','user','birthday'),
    ('更正一下，我这周的生日是七月八日。','我这周的生日是七月八日。','user','birthday'),
    ('“更正一下，我的生日是七月八日。”是台词。','我的生日是七月八日。','user','birthday'),
    ('更正一下，我的生日是七月八日，开玩笑的。','我的生日是七月八日','user','birthday'),
    ('更正一下，我的生日是七月八日。','我的生日是七月八日。','user','job'),
    ('更正一下，我的生日是七月八日。','我的生日是七月八日。','user','other_owner'),
    ('更正一下，我姐姐的生日是七月八日。','我姐姐的生日是七月八日。','related:我姐姐','birthday'),
    ('更正一下我的职业，我的生日是七月八日。','我的生日是七月八日。','user','birthday'),
    ('朋友说，更正一下，我的生日是七月八日。','我的生日是七月八日。','user','birthday'),
    ('朋友说，我的生日改为七月八日。','我的生日改为七月八日。','user','birthday'),
])
def test_unverified_correction_cannot_bypass_admission(text,evidence,subject,target,tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    birthday,job=prepare(store)
    foreign=store.observe('bob','one','我的生日是三月四日',[statement('我的生日是三月四日')])['saved'][0]
    ident={'birthday':birthday,'job':job,'other_owner':foreign}[target]
    plan=dict(utterance_kind='self_report',memory_scores=[],queries=[],forget_ids=[],candidates=[dict(
        subject=subject,predicate='statement',value=evidence.rstrip('。'),evidence=evidence,polarity='positive',
        degree='ordinary',importance=.8,stability=.2,supersedes=ident)])
    before=store.records('alice',active_only=False)
    def generated(messages,*,kind):return json.dumps(plan,ensure_ascii=False) if kind=='plan' else 'fixture'
    Runtime(store,generated).reply('alice','attempt',[dict(role='user',content=text)])
    assert store.records('alice',active_only=False)==before
    assert len(store.records('bob'))==1


def test_new_weak_candidate_and_low_confidence_correction_still_rejected(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db');birthday,_=prepare(store)
    c=statement('我的生日是七月八日。',stability=.2,supersedes=birthday,confidence=.5)
    assert store.observe('alice','weak','更正一下，我的生日是七月八日。',[c])['rejected']==1
    c=statement('我的学校是星河学院。',stability=.2)
    assert store.observe('alice','new','请记住我的学校是星河学院。',[c])['rejected']==1
    c=statement('我的生日是七月八日。',stability=.2)
    assert store.observe('alice','missing-target','更正一下，我的生日是七月八日。',[c])['rejected']==1
    assert len(store.records('alice'))==2


def test_in_clause_attribute_change_has_a_scoped_target(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db');birthday,job=prepare(store)
    text='我的生日改为七月八日。'
    c=statement(text,stability=.2,supersedes=birthday)
    assert store.observe('alice','change',text,[c])['status']=='saved'
    assert {r['value'] for r in store.records('alice')}=={'我的生日改为七月八日','我的职业是园丁'}


def test_word_containing_speech_character_is_not_a_reported_correction(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    old_text='我的职业是小说编辑'
    old=store.observe('alice','one',old_text,[statement(old_text)])['saved'][0]
    c=statement('我的职业是说书人。',stability=.2,supersedes=old)
    assert store.observe('alice','two','更正一下，我的职业是说书人。',[c])['status']=='saved'
    assert store.records('alice')[0]['value']=='我的职业是说书人'


@pytest.mark.parametrize('old_text,new_text,predicate,old_value,new_value,polarity,degree',[
    ('我叫小青','我叫小桐','name','小青','小桐','positive','ordinary'),
    ('我喜欢葡萄干','我不喜欢葡萄干','preference','葡萄干','葡萄干','negative','ordinary'),
    ('我喜欢红豆糕','我最喜欢红豆糕','preference','红豆糕','红豆糕','positive','favorite'),
    ('我对鸡蛋过敏','我对鸡蛋不过敏','allergy','鸡蛋','鸡蛋','negative','ordinary'),
])
def test_direct_typed_corrections_keep_literal_meaning(old_text,new_text,predicate,old_value,new_value,polarity,degree,tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    old=store.observe('alice','one',old_text)['saved'][0]
    c=Candidate('user',predicate,new_value,new_text,polarity=polarity,degree=degree,stability=.2)
    assert store.observe('alice','two',new_text,[c])['status']=='saved'
    row=store.records('alice')[0]
    assert (row['value'],row['polarity'],row['degree'],row['supersedes'])==(new_value,polarity,degree,old)


def test_correcting_a_fact_cannot_partially_invalidate_it_if_write_fails(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db');birthday,_=prepare(store)
    before=store.records('alice',active_only=False)
    with store.connect() as db:
        db.execute("CREATE TRIGGER fail_correct BEFORE INSERT ON facts BEGIN SELECT RAISE(ABORT,'disk failure'); END")
    c=statement('我的生日是七月八日。',stability=.2,supersedes=birthday)
    with pytest.raises(sqlite3.IntegrityError,match='disk failure'):
        store.observe('alice','failed','更正一下，我的生日是七月八日。',[c])
    assert store.records('alice',active_only=False)==before
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM processed WHERE source_id='failed'").fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM fact_corrections').fetchone()[0]==0


def test_cross_turn_quote_classification_blocks_otherwise_valid_correction(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db');birthday,_=prepare(store)
    before=store.records('alice',active_only=False)
    plan=dict(utterance_kind='quotation',candidates=[dict(subject='user',predicate='statement',
        value='我的生日是七月八日',evidence='我的生日是七月八日。',polarity='positive',degree='ordinary',
        importance=.8,stability=.2,supersedes=birthday)],memory_scores=[],queries=[],forget_ids=[])
    def generated(messages,*,kind):return json.dumps(plan,ensure_ascii=False) if kind=='plan' else 'fixture'
    Runtime(store,generated).reply('alice','quote',[
        dict(role='user',content='以下是人物台词。'),dict(role='assistant',content='好。'),
        dict(role='user',content='更正一下，我的生日是七月八日。')])
    assert store.records('alice',active_only=False)==before
