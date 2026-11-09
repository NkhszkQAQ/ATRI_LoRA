"""Source-checked, stage-scoped canonical retrieval using the existing reranker.

Stage filtering happens before ranking; later scenes never compete for an early
character's knowledge. Personal memory decay does not apply to this corpus.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading

from .contracts import STAGES


def validate_dense_settings(model, threshold):
    """Dense scores need their own explicit, corpus-calibrated threshold."""
    if model is None and threshold is None:
        return
    if (not isinstance(model, str) or not model.strip()
            or type(threshold) not in (int, float)
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('dense model and calibrated threshold in [0, 1] are required together')


class LocalDenseScorer:
    """CPU BGE query/document similarity; never a factual confidence score."""
    def __init__(self, model):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device='cpu', trust_remote_code=False)
        self.model.max_seq_length = 512
        self._cache = {}
        self._lock = threading.Lock()

    def score(self, query, texts):
        if not texts:
            return []
        key = tuple(texts)
        with self._lock:
            if key not in self._cache:
                # Only small, stage-filtered canonical card sets are cached.
                if len(self._cache) >= 8:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = self.model.encode(list(key), normalize_embeddings=True,
                    batch_size=16, show_progress_bar=False)
            vector = self.model.encode(['为这个句子生成表示以用于检索相关文章：' + query],
                normalize_embeddings=True, show_progress_bar=False)[0]
            return [float(v) for v in self._cache[key] @ vector]

class StageCorpus:
    def __init__(self, source, card_files, stage, *, reranker, threshold=0.5, top_k=3,
                 dense_scorer=None, dense_threshold=None):
        if stage not in STAGES or not 0 <= threshold <= 1 or type(top_k) is not int or top_k < 1:
            raise ValueError('invalid story retrieval settings')
        self.stage, self.reranker, self.threshold, self.top_k = stage, reranker, threshold, top_k
        validate_dense_settings('configured' if dense_scorer is not None else None, dense_threshold)
        self.dense_scorer, self.dense_threshold = dense_scorer, dense_threshold
        root, self.cards = Path(source).resolve(strict=True), []
        seen, verified = set(), {}
        for file in card_files:
            collection = json.loads(Path(file).read_text())
            if collection.get('version') != 1:
                raise ValueError('unsupported canonical collection')
            for card in collection['cards']:
                if card['id'] in seen:
                    raise ValueError('duplicate canonical ID')
                seen.add(card['id'])
                stages = card.get('allowed_stages')
                # Existing early_story cards retain their expressly limited scope.
                if stages is None and card.get('scope') == 'early_story':
                    stages = ['daily']
                if not isinstance(stages, list) or not stages or any(s not in STAGES for s in stages):
                    raise ValueError('explicit known story stages required')
                if not card.get('evidence') or not card.get('answer'):
                    raise ValueError('canonical answer and evidence required')
                for ref in card['evidence']:
                    path = (root / ref['source']).resolve(strict=True)
                    if not path.is_relative_to(root):
                        raise ValueError('canonical source escapes source directory')
                    if path not in verified:
                        raw = path.read_bytes()
                        verified[path] = (hashlib.sha256(raw).hexdigest(), json.loads(raw))
                    sha, scenes = verified[path]
                    if sha != ref['sha256']:
                        raise ValueError('canonical source SHA mismatch')
                    if not 1 <= ref['scene'] <= len(scenes):
                        raise ValueError('invalid canonical scene')
                    dialogues = scenes[ref['scene']-1]['dialogues']
                    if not 1 <= ref['dialogue_start'] <= ref['dialogue_end'] <= len(dialogues):
                        raise ValueError('invalid canonical dialogue range')
                if stage in stages:
                    self.cards.append({**card, 'allowed_stages': stages})
        self.source_hashes = {str(p.relative_to(root)): value[0] for p, value in verified.items()}

    def query_topics(self):
        """Verified index headings, not answer evidence or unfiltered facts."""
        return [{'title': card['title'], 'allowed_stages': list(card['allowed_stages'])}
                for card in self.cards]

    def search(self, query, history=()):
        """Primary retrieval for generated queries and combined history."""
        return self._search(query, allow_dense=False)

    def search_current(self, query, history=()):
        """Supplement only the actual current utterance used in calibration."""
        return self._search(query, allow_dense=True)

    def _search(self, query, *, allow_dense):
        if not isinstance(query, str) or not query.strip() or not self.cards:
            return []
        # Real utterances/context and optional planner rewrites share this path;
        # no punctuation, fixed entity, or question-word gate is used here.
        texts = [card['title'] + '\n' + card['answer'] + '\n' + '；'.join(card.get('questions', [])) for card in self.cards]
        scores = self.reranker.score(query, texts)
        if len(scores) != len(texts):
            raise ValueError('reranker score count mismatch')
        hits = []
        for card, text, score in zip(self.cards, texts, scores):
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('invalid reranker score')
            if score >= self.threshold:
                hits.append({'node_id': 'canonical:' + card['id'], 'score': float(score),
                    'retrieval_query': query, 'text': card['title'] + '\n' + card['answer'],
                    'metadata': {'kind': 'canonical_summary', 'source': 'facts/' + card['id'],
                                 'allowed_stages': card['allowed_stages'], 'evidence': card['evidence'],
                                 'event_time': card.get('event_time', 'see_citation'),
                                 'revealed_at': card.get('revealed_at', 'see_citation'),
                                 'knowledge_state': card.get('knowledge_state', card.get('scope'))}})
        hits = sorted(hits, key=lambda hit: -hit['score'])[:self.top_k]
        if not allow_dense or self.dense_scorer is None or len(hits) == self.top_k:
            return hits
        dense_scores = self.dense_scorer.score(query, texts)
        if len(dense_scores) != len(texts):
            raise ValueError('dense score count mismatch')
        if any(type(s) not in (int, float) or not math.isfinite(s) or not -1 <= s <= 1
               for s in dense_scores):
            raise ValueError('invalid dense score')
        selected = {hit['node_id'] for hit in hits}
        for i in sorted(range(len(self.cards)), key=lambda i: -dense_scores[i]):
            card, score = self.cards[i], dense_scores[i]
            ident = 'canonical:' + card['id']
            if score < self.dense_threshold or ident in selected:
                continue
            hits.append({'node_id': ident, 'score': score, 'retrieval_tier': 1,
                'retrieval_query': query, 'text': card['title'] + '\n' + card['answer'],
                'metadata': {'kind': 'canonical_summary', 'source': 'facts/' + card['id'],
                    'allowed_stages': card['allowed_stages'], 'evidence': card['evidence'],
                    'event_time': card.get('event_time', 'see_citation'),
                    'revealed_at': card.get('revealed_at', 'see_citation'),
                    'knowledge_state': card.get('knowledge_state', card.get('scope'))}})
            if len(hits) == self.top_k:
                break
        # Reranker admissions keep precedence; cosine scores never compete
        # numerically with their scores, including across runtime queries.
        return hits
