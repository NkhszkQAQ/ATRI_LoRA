"""Source-checked, stage-scoped canonical retrieval using the existing reranker.

Stage filtering happens before ranking; later scenes never compete for an early
character's knowledge. Personal memory decay does not apply to this corpus.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from atri_runtime import STAGES


class StageCorpus:
    def __init__(self, source, card_files, stage, *, reranker, threshold=0.5, top_k=3):
        if stage not in STAGES or not 0 <= threshold <= 1 or type(top_k) is not int or top_k < 1:
            raise ValueError('invalid story retrieval settings')
        self.stage, self.reranker, self.threshold, self.top_k = stage, reranker, threshold, top_k
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

    def search(self, query, history=()):
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
        return sorted(hits, key=lambda hit: -hit['score'])[:self.top_k]
