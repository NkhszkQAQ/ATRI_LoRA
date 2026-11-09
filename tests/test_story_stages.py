import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atri_story import StageCorpus, validate_dense_settings


@pytest.fixture
def corpus_files(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    raw = json.dumps([{'dialogues': [{'character': '角色', 'text': '测试原文'}]}]).encode()
    (source/'scene.txt').write_bytes(raw)
    ref = {'source': 'scene.txt', 'sha256': hashlib.sha256(raw).hexdigest(),
           'scene': 1, 'dialogue_start': 1, 'dialogue_end': 1}
    cards = [{'id': stage, 'title': stage, 'answer': stage, 'allowed_stages': [stage], 'evidence': [ref]}
             for stage in ['daily', 'eden_partial', 'past_recovered']]
    target = tmp_path/'cards.json'
    target.write_text(json.dumps({'version': 1, 'cards': cards}))
    return source, target


def test_filtering_precedes_model_ranking(corpus_files):
    source, target = corpus_files
    seen = []
    class Ranker:
        def score(self, query, texts):
            seen.extend(texts)
            return [0.8]*len(texts)
    corpus = StageCorpus(source, [target], 'daily', reranker=Ranker())
    hits = corpus.search('一起聊聊过去的事')
    assert len(hits) == 1 and hits[0]['node_id'] == 'canonical:daily'
    assert not any('eden_partial' in t or 'past_recovered' in t for t in seen)


def test_retrieval_never_modifies_sources_and_checks_hash(corpus_files):
    source, target = corpus_files
    before = (source/'scene.txt').read_bytes()
    class Ranker:
        def score(self, query, texts):
            return [0.1]*len(texts)
    corpus = StageCorpus(source, [target], 'daily', reranker=Ranker())
    assert corpus.search('毫不相关的话') == []
    assert (source/'scene.txt').read_bytes() == before
    (source/'scene.txt').write_bytes(before+b' ')
    with pytest.raises(ValueError, match='SHA'):
        StageCorpus(source, [target], 'daily', reranker=Ranker())


def test_invalid_scope_fails_closed(corpus_files):
    source, target = corpus_files
    data = json.loads(target.read_text())
    del data['cards'][0]['allowed_stages']
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='stages'):
        StageCorpus(source, [target], 'daily', reranker=None)


def test_dense_supplements_without_replacing_primary_or_crossing_stage(corpus_files):
    source, target = corpus_files
    data = json.loads(target.read_text())
    data['cards'].extend([{**data['cards'][0], 'id': ident, 'title': ident}
                          for ident in ['second', 'third']])
    target.write_text(json.dumps(data))
    class Primary:
        def score(self, query, texts):
            return [.6, .1, .2]
    seen = []
    class Dense:
        def score(self, query, texts):
            seen.extend(texts)
            return [.7, .95, .8]
    hits = StageCorpus(source, [target], 'daily', reranker=Primary(), top_k=2,
        dense_scorer=Dense(), dense_threshold=.75).search_current('提出一个相关活动')
    assert [h['node_id'] for h in hits] == ['canonical:daily', 'canonical:second']
    assert 'retrieval_tier' not in hits[0]
    assert hits[1]['retrieval_tier'] == 1
    assert all('eden_partial' not in text and 'past_recovered' not in text for text in seen)


@pytest.mark.parametrize('values', [[float('nan')], [1.1], [], [.8, .9]])
def test_bad_dense_scores_are_rejected(corpus_files, values):
    source, target = corpus_files
    class Primary:
        def score(self, query, texts): return [.1]
    class Dense:
        def score(self, query, texts): return values
    corpus = StageCorpus(source, [target], 'daily', reranker=Primary(),
                         dense_scorer=Dense(), dense_threshold=.5)
    with pytest.raises(ValueError, match='dense score'):
        corpus.search_current('相关话题')


def test_dense_cannot_admit_below_its_calibrated_threshold(corpus_files):
    source, target = corpus_files
    class Primary:
        def score(self, query, texts): return [.1]
    class Dense:
        def score(self, query, texts): return [.49]
    corpus = StageCorpus(source, [target], 'daily', reranker=Primary(),
                         dense_scorer=Dense(), dense_threshold=.5)
    assert corpus.search_current('无关话题') == []
    assert corpus.search_current('') == []


def test_generated_query_does_not_receive_uncalibrated_dense_admission(corpus_files):
    source, target = corpus_files
    class Primary:
        def score(self, query, texts): return [.1]
    class Dense:
        def score(self, query, texts): return [.8]
    corpus = StageCorpus(source, [target], 'daily', reranker=Primary(),
                         dense_scorer=Dense(), dense_threshold=.5)
    assert corpus.search('改写后的查询') == []
    assert len(corpus.search_current('用户原话')) == 1


@pytest.mark.parametrize('model,threshold', [
    ('model', None), (None, .5), ('', .5), ('model', float('nan')),
    ('model', 1.1), ('model', -.1), ('model', True),
])
def test_dense_configuration_is_explicit_and_validated(model, threshold):
    with pytest.raises(ValueError, match='required together'):
        validate_dense_settings(model, threshold)
