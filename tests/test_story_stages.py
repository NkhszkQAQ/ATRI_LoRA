import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atri_story import StageCorpus


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
