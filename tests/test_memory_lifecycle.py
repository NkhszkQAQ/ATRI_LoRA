import concurrent.futures
import json
import os
from dataclasses import replace
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atri_memory_lifecycle import Candidate, LifecycleStore, Policy, extract_explicit


def test_separate_process_restart_preserves_correction_and_delete_tombstone(tmp_path):
    database = tmp_path/'restart.db'
    def run(body):
        script = ('import json, os, sys\nfrom atri_memory_lifecycle import LifecycleStore\n'
                  'store=LifecycleStore(sys.argv[1])\n' + body)
        result = subprocess.run([sys.executable, '-c', script, str(database)], check=True,
            text=True, capture_output=True, env={**os.environ,
            'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'tools')})
        return json.loads(result.stdout)
    first = run("store.observe('alice','save','我喜欢咖啡')\n"
                "store.observe('bob','save','我喜欢咖啡')\n"
                "print(json.dumps({'pid':os.getpid(),'rows':store.records('alice')}))")
    corrected = run("assert store.records('alice')[0]['polarity']=='positive'\n"
                    "store.observe('alice','correction','我不喜欢咖啡')\n"
                    "print(json.dumps({'pid':os.getpid(),'rows':store.records('alice')}))")
    deleted = run("rows=store.records('alice')\n"
                  "assert len(rows)==1 and rows[0]['polarity']=='negative'\n"
                  "store.forget_event('alice','forget','忘记咖啡',[rows[0]['id']])\n"
                  "print(json.dumps({'pid':os.getpid(),'rows':store.records('alice',active_only=False)}))")
    restarted = run("assert store.observe('alice','save','我喜欢咖啡')['status']=='duplicate'\n"
                    "assert store.observe('alice','correction','我不喜欢咖啡')['status']=='duplicate'\n"
                    "print(json.dumps({'pid':os.getpid(),'alice':store.records('alice',active_only=False),'bob':store.records('bob')}))")
    assert len({r['pid'] for r in [first,corrected,deleted,restarted]}) == 4
    assert first['rows'][0]['polarity'] == 'positive'
    assert corrected['rows'][0]['supersedes'] == first['rows'][0]['id']
    assert deleted['rows'] == restarted['alice'] == []
    assert restarted['bob'][0]['polarity'] == 'positive'


@pytest.mark.parametrize('text', [
    '朋友说我喜欢咖啡', '他说：“我喜欢咖啡”', '我喜欢咖啡吗？',
    '我今天喜欢喝热水', '我喜欢咖啡，开玩笑的', '假如我喜欢咖啡',
    '比如我喜欢咖啡', '我喜欢咖啡才怪', '我喜欢你说的那句话',
    '我们出去走走吧', '我这次选热水', '建议你试试咖啡',
])
def test_ambiguous_or_temporary_is_not_automatically_saved(text):
    assert extract_explicit(text) == []


def test_subject_negation_degree_and_single_important_fact(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    store.observe('alice', 'one', '我的朋友喜欢抹茶，我不喜欢咖啡，我最喜欢蓝色，我对花生过敏')
    rows = store.records('alice')
    assert {(r['subject'], r['value'], r['polarity'], r['degree']) for r in rows} == {
        ('related:我的朋友', '抹茶', 'positive', 'ordinary'),
        ('user', '咖啡', 'negative', 'ordinary'),
        ('user', '蓝色', 'positive', 'favorite'),
        ('user', '花生', 'positive', 'ordinary'),
    }
    assert next(r for r in rows if r['predicate'] == 'allergy')['importance'] == 1
    assert store.records('bob') == []


def test_restart_correction_and_deletion_do_not_restore_history(tmp_path):
    path = tmp_path / 'facts.db'
    store = LifecycleStore(path)
    first = store.observe('alice', '1', '我喜欢咖啡')['saved'][0]
    store.observe('bob', '1', '我喜欢咖啡')
    store.observe('alice', '2', '我不喜欢咖啡')
    store = LifecycleStore(path)
    assert len(store.records('alice')) == 1
    assert store.records('alice')[0]['supersedes'] == first
    assert store.records('alice')[0]['polarity'] == 'negative'
    assert len(store.records('alice', active_only=False)) == 2
    assert store.forget('alice', value='咖啡')['count'] == 2
    assert store.observe('alice', '1', '我喜欢咖啡')['status'] == 'duplicate'
    assert store.records('alice', active_only=False) == []
    assert len(store.records('bob')) == 1


def test_multiple_likes_survive_name_correction(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    store.observe('alice', '1', '我喜欢画画，我喜欢散步，我叫小林')
    store.observe('alice', '2', '我叫小雨')
    assert {r['value'] for r in store.records('alice')} == {'画画', '散步', '小雨'}


@pytest.mark.parametrize('select_old,use_event', [(False, True), (True, True), (False, False)])
def test_forget_clears_cross_value_correction_history_and_preserves_scope(select_old, use_event, tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    first = store.observe('alice', 'one', '我喜欢红色')['saved'][0]
    store.observe('alice', 'unrelated', '我喜欢画画，我的朋友喜欢红色')
    store.observe('bob', 'one', '我喜欢红色')
    second = store.observe('alice', 'two', '我把喜欢的颜色改成蓝色', [
        Candidate('user', 'preference', '蓝色', '我把喜欢的颜色改成蓝色', supersedes=first)])['saved'][0]
    third = store.observe('alice', 'three', '我把喜欢的颜色改成绿色', [
        Candidate('user', 'preference', '绿色', '我把喜欢的颜色改成绿色', supersedes=second)])['saved'][0]
    if use_event:
        result = store.forget_event('alice', 'forget', '忘记颜色', [first if select_old else third])
    else:
        result = store.forget('alice', subject='user', predicate='preference', value='绿色')
    assert result['count'] == 3
    assert {(r['subject'], r['value']) for r in store.records('alice', active_only=False)} == {
        ('user', '画画'), ('related:我的朋友', '红色')}
    assert store.records('bob')[0]['value'] == '红色'
    assert store.observe('alice', 'one', '我喜欢红色')['status'] == 'duplicate'
    assert all(r['subject'] != 'user' or r['value'] != '红色' for r in store.records('alice', active_only=False))


def test_deletion_family_handles_overlapping_targets_and_cycles_once(tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    first = store.observe('alice', 'one', '我叫小林')['saved'][0]
    second = store.observe('alice', 'two', '我叫小雨')['saved'][0]
    # Corrupt historical links must not cause an infinite traversal or escape
    # the owner boundary; real correction insertion creates only older links.
    other = store.observe('bob', 'one', '我叫小林')['saved'][0]
    with store.connect() as db:
        db.execute('UPDATE facts SET supersedes=? WHERE id=?', (second, first))
        db.execute('UPDATE facts SET supersedes=? WHERE id=?', (first, other))
    assert store.forget_event('alice', 'delete', '忘记名字', [first, second])['count'] == 2
    assert not store.records('alice', active_only=False)
    assert store.records('bob')[0]['id'] == other


def test_correction_to_existing_fact_preserves_both_histories_for_deletion(tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    red = store.observe('alice', 'one', '我喜欢红色')['saved'][0]
    blue = store.observe('alice', 'two', '我喜欢蓝色')['saved'][0]
    green = store.observe('alice', 'three', '我把喜欢的颜色改成绿色', [
        Candidate('user', 'preference', '绿色', '我把喜欢的颜色改成绿色', supersedes=blue)])['saved'][0]
    store.observe('alice', 'four', '我把喜欢的颜色改成绿色', [
        Candidate('user', 'preference', '绿色', '我把喜欢的颜色改成绿色', supersedes=red)])
    store = LifecycleStore(tmp_path/'facts.db')
    assert store.forget_event('alice', 'delete', '忘记颜色', [green])['count'] == 3
    assert not store.records('alice', active_only=False)
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM fact_corrections').fetchone()[0] == 0


def test_existing_lifecycle_database_gains_links_without_rewriting_old_rows(tmp_path):
    path = tmp_path/'facts.db'
    store = LifecycleStore(path)
    store.observe('alice', 'one', '我叫小林')
    store.observe('alice', 'two', '我叫小雨')
    before = store.records('alice', active_only=False)
    with store.connect() as db:
        db.execute('DROP TABLE fact_corrections')  # Simulate pre-extension schema.
    store = LifecycleStore(path)
    assert store.records('alice', active_only=False) == before
    assert store.forget('alice', value='小雨')['count'] == 2  # Legacy links still work.


def test_explicit_correction_is_scoped_and_keeps_other_preferences(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    store.observe('alice', '1', '我喜欢红色，我喜欢画画')
    old = next(r for r in store.records('alice') if r['value'] == '红色')
    c = Candidate('user', 'preference', '蓝色', '我把喜欢的颜色改成蓝色', supersedes=old['id'])
    store.observe('alice', '2', c.evidence, [c])
    assert {r['value'] for r in store.records('alice')} == {'蓝色', '画画'}
    with pytest.raises(ValueError, match='same owner'):
        store.observe('bob', '2', c.evidence, [c])
    assert not store.records('bob')


def test_negative_sentence_particle_does_not_create_another_object(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    store.observe('alice', '1', '我喜欢咖啡')
    store.observe('alice', '2', '我现在不喜欢咖啡了')
    assert [(r['value'], r['polarity']) for r in store.records('alice')] == [('咖啡', 'negative')]


def test_concurrent_event_is_counted_once(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(lambda _: store.observe('alice', '1', '我喜欢画画'), range(8)))
    assert sum(r['status'] == 'saved' for r in receipts) == 1
    assert store.records('alice')[0]['confirmations'] == 1
    store.observe('alice', '2', '我喜欢画画')
    assert store.records('alice')[0]['confirmations'] == 2
    with pytest.raises(ValueError, match='reused'):
        store.observe('alice', '2', '我喜欢散步')


def test_repeated_clauses_count_once_per_record_per_user_event(tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    first = store.observe('alice', 'one', '我喜欢画画，我喜欢画画，我喜欢画画')
    ident = first['saved'][0]
    assert first['saved'] == [ident] and first['confirmed'] == []
    assert store.records('alice')[0]['confirmations'] == 1
    second = store.observe('alice', 'two', '我喜欢画画，请记住我喜欢画画')
    assert second['saved'] == [] and second['confirmed'] == [ident]
    row = store.records('alice')[0]
    assert row['confirmations'] == 2 and row['pinned'] == 1
    assert store.observe('alice', 'two', '我喜欢画画，请记住我喜欢画画')['status'] == 'duplicate'
    assert store.records('alice')[0]['confirmations'] == 2


def test_distinct_facts_and_later_polarity_changes_are_not_deduplicated(tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    store.observe('alice', 'one', '我喜欢画画，我喜欢散步，我不喜欢画画，我喜欢画画')
    assert {(r['value'], r['polarity']) for r in store.records('alice')} == {
        ('画画', 'positive'), ('散步', 'positive')}
    assert len(store.records('alice', active_only=False)) == 4
    assert all(r['confirmations'] == 1 for r in store.records('alice', active_only=False))


def test_ranking_decay_confirmation_and_pinned_retention(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db', clock=lambda: 100)
    store.observe('alice', '1', '请记住我喜欢画画，我对花生过敏')
    score = lambda rows: [0.9 for _ in rows]
    day0 = store.rank('alice', score, now=100)
    day30 = store.rank('alice', score, now=100+30*86400)
    day60 = store.rank('alice', score, now=100+60*86400)
    assert all(a['score'] > b['score'] > c['score'] for a, b, c in zip(day0, day30, day60))
    assert all(r['recency'] == pytest.approx(0.5) for r in day30)
    assert store.rank('alice', lambda rows: [0 for _ in rows], now=100) == []
    assert all(r['confirmations'] == 1 for r in store.records('alice'))
    store.observe('alice', '2', '我喜欢画画')
    repeated = next(r for r in store.rank('alice', score, now=100+30*86400) if r['value'] == '画画')
    assert repeated['recency'] > 0.5
    store.rank('alice', score, now=100+10000*86400)
    assert len(store.records('alice')) == 2
    assert next(r for r in store.records('alice') if r['value'] == '画画')['pinned']


def test_admission_evidence_and_atomic_rollback(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db')
    c = Candidate('user', 'preference', '画画', '我喜欢画画')
    with pytest.raises(ValueError, match='cite'):
        store.observe('alice', '1', '他说你好', [c])
    assert store.observe('alice', '2', c.evidence, [replace(c, stability=0.2)])['rejected'] == 1
    # Simulate an actual SQLite failure during the transaction, after candidate
    # validation. Neither a success receipt nor a processed event may survive.
    with store.connect() as db:
        db.execute("CREATE TRIGGER fail_write BEFORE INSERT ON facts BEGIN SELECT RAISE(ABORT,'disk failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='disk failure'):
        store.observe('alice', '3', c.evidence)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM processed WHERE source_id='3'").fetchone()[0] == 0
    assert store.records('alice') == []


def test_ranking_only_passes_authenticated_owner_rows_and_enforces_budget(tmp_path):
    store = LifecycleStore(tmp_path / 'facts.db', Policy(context_chars=128))
    store.observe('alice', '1', '我喜欢画画')
    store.observe('bob', '1', '我喜欢唱歌')
    seen = []
    def relevance(rows):
        seen.extend(rows)
        return [1 for _ in rows]
    assert store.rank('alice', relevance) == []
    assert {r['owner'] for r in seen} == {'alice'}
    with pytest.raises(ValueError, match='relevance'):
        store.rank('alice', lambda rows: [float('nan')])


@pytest.mark.parametrize('options', [
    {'half_life_days': 0}, {'half_life_days': float('nan')},
    {'admission': 2}, {'limit': True}, {'context_chars': 0},
    {'confirmation_gain': -1}, {'minimum_relevance': float('inf')},
])
def test_policy_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        Policy(**options)


def test_legacy_database_is_rejected_without_changing_its_contents(tmp_path):
    path = tmp_path/'legacy.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE memories(owner TEXT, value TEXT)')
        db.execute("INSERT INTO memories VALUES('fictional','preserved')")
    with pytest.raises(ValueError, match='separate'):
        LifecycleStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT * FROM memories').fetchall() == [('fictional','preserved')]
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [('memories',)]


def test_delete_event_replay_cannot_remove_later_reintroduced_fact(tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    old = store.observe('alice','1','我喜欢画画')['saved'][0]
    assert store.forget_event('alice','2','忘记画画',[old])['count'] == 1
    new = store.observe('alice','3','我喜欢画画')['saved'][0]
    assert store.forget_event('alice','2','忘记画画',[new])['status'] == 'duplicate'
    with pytest.raises(ValueError,match='reused'):
        store.forget_event('alice','3','忘记画画',[new])
    assert store.records('alice')[0]['id'] == new


def test_delete_event_is_owner_scoped_and_atomic(tmp_path):
    store = LifecycleStore(tmp_path/'facts.db')
    a = store.observe('alice','1','我喜欢画画')['saved'][0]
    b = store.observe('bob','1','我喜欢唱歌')['saved'][0]
    with pytest.raises(ValueError,match='owner'):
        store.forget_event('alice','2','忘记这些',[a,b])
    assert store.records('alice')[0]['id'] == a
    with store.connect() as db:
        db.execute("CREATE TRIGGER fail_delete BEFORE DELETE ON facts BEGIN SELECT RAISE(ABORT,'failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.forget_event('alice','2','忘记这些',[a])
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM processed WHERE owner='alice' AND source_id='2'").fetchone()[0] == 0
    assert store.records('alice')[0]['id'] == a
