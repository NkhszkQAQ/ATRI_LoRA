"""Owner-scoped memory lifecycle. No model, network, or production DB defaults.

Extraction, admission, ranking, invalidation, and deletion are separate operations.
The caller supplies an authenticated owner; candidate data cannot choose an owner.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid


@dataclass(frozen=True)
class Policy:
    admission: float = 0.8
    minimum_relevance: float = 0.35
    relevance_weight: float = 0.65
    importance_weight: float = 0.2
    recency_weight: float = 0.15
    half_life_days: float = 30.0
    confirmation_gain: float = 0.5
    max_strength: float = 4.0
    limit: int = 6
    context_chars: int = 3500

    def __post_init__(self):
        for key in ('admission', 'minimum_relevance', 'relevance_weight',
                    'importance_weight', 'recency_weight'):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'invalid {key}')
        if self.relevance_weight + self.importance_weight + self.recency_weight <= 0:
            raise ValueError('ranking needs a positive weight')
        for key in ('half_life_days', 'max_strength'):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid {key}')
        if not math.isfinite(self.confirmation_gain) or self.confirmation_gain < 0:
            raise ValueError('invalid confirmation_gain')
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError('invalid limit')
        if type(self.context_chars) is not int or self.context_chars < 128:
            raise ValueError('invalid context budget')


@dataclass(frozen=True)
class Candidate:
    subject: str
    predicate: str
    value: str
    evidence: str
    polarity: str = 'positive'
    degree: str = 'ordinary'
    confidence: float = 1.0
    importance: float = 0.5
    stability: float = 1.0
    explicit_keep: bool = False
    supersedes: str | None = None

    def validate(self, source):
        if not isinstance(self.subject, str) or (self.subject != 'user' and (
                not self.subject.startswith('related:') or not self.subject.removeprefix('related:').strip())):
            raise ValueError('candidate subject must identify the user or a related person')
        for key in ('subject', 'predicate', 'value', 'evidence'):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip() or len(value) > 400:
                raise ValueError(f'invalid candidate {key}')
        if self.evidence not in source:
            raise ValueError('candidate must cite current user text verbatim')
        if self.polarity not in {'positive', 'negative'} or self.degree not in {'ordinary', 'favorite'}:
            raise ValueError('invalid polarity or degree')
        for value in (self.confidence, self.importance, self.stability):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid candidate score')
        if type(self.explicit_keep) is not bool:
            raise ValueError('invalid retention flag')
        if self.supersedes is not None and (not isinstance(self.supersedes, str) or not self.supersedes):
            raise ValueError('invalid correction ID')


def extract_explicit(text):
    """Conservative admission path for unambiguous self-reports.

    Ambiguous, quoted, joking, and temporary claims are left to dialogue, never
    promoted by a broad search for the word 'like'. This is not semantic coverage.
    """
    if not isinstance(text, str) or len(text) > 12000:
        return []
    if re.search(r'[“”「」『』\"？?]|开玩笑|逗你|假如|假设|比如|例如|台词|如果|才怪', text):
        return []
    candidates = []
    for clause in re.split(r'[，,。！!；;\n]+', text):
        clause = clause.strip()
        if not clause or re.search(r'今天|这次|暂时|临时|刚才|现在想|一会儿', clause):
            continue
        match = re.fullmatch(
            r'(?P<keep>请记住|记住|别忘了)?(?P<who>我(?:的(?:朋友|妈妈|爸爸|姐姐|哥哥|同学))?)'
            r'(?P<time>一直|平时|通常|现在|以后)?(?P<op>最喜欢|不再喜欢|不喜欢|讨厌|喜欢|叫|对)'
            r'(?P<value>[^，,。！!；;\n]{1,60})', clause)
        if not match:
            continue
        who, op, value = match['who'], match['op'], match['value'].strip()
        if op in {'不再喜欢', '不喜欢', '讨厌'}:
            value = value.removesuffix('了')
        if re.search(r'说|告诉|觉得|可能|也许|应该|不一定|但是|不过|而是|并非|不是|才怪', value):
            continue
        subject = 'user' if who == '我' else 'related:' + who
        predicate, importance, degree = 'preference', 0.5, 'ordinary'
        if op == '叫':
            predicate, importance = 'name', 0.7
        elif op == '对':
            if not value.endswith('过敏'):
                continue
            predicate, value, importance = 'allergy', value[:-2].strip(), 1.0
        elif op == '最喜欢':
            degree = 'favorite'
        if not value:
            continue
        # Do not guess a food/drink taxonomy. Preserve the actual reported value.
        candidates.append(Candidate(subject, predicate, value, clause,
            polarity='negative' if op in {'不再喜欢', '不喜欢', '讨厌'} else 'positive',
            degree=degree, importance=importance, explicit_keep=bool(match['keep'])))
    return candidates


class LifecycleStore:
    def __init__(self, path, policy=None, *, clock=time.time):
        self.path, self.policy, self.clock = str(path), policy or Policy(), clock
        Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        with self.connect() as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if tables - {'facts', 'processed'}:
                raise ValueError('use a separate lifecycle database; legacy/other tables found')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, subject TEXT NOT NULL,
                    predicate TEXT NOT NULL, value TEXT NOT NULL, polarity TEXT NOT NULL,
                    degree TEXT NOT NULL, evidence TEXT NOT NULL, source_id TEXT NOT NULL,
                    created REAL NOT NULL, confirmed REAL NOT NULL,
                    confirmations INTEGER NOT NULL, confidence REAL NOT NULL,
                    importance REAL NOT NULL, stability REAL NOT NULL, pinned INTEGER NOT NULL,
                    status TEXT NOT NULL, supersedes TEXT);
                CREATE INDEX IF NOT EXISTS facts_owner_state ON facts(owner,status);
                CREATE TABLE IF NOT EXISTS processed (
                    owner TEXT NOT NULL, source_id TEXT NOT NULL, digest TEXT NOT NULL,
                    PRIMARY KEY(owner,source_id));
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _owner(owner):
        if not isinstance(owner, str) or not owner.strip() or len(owner) > 220:
            raise ValueError('authenticated owner required')

    def records(self, owner, *, active_only=True):
        self._owner(owner)
        with self.connect() as db:
            rows = db.execute('SELECT * FROM facts WHERE owner=?' +
                (" AND status='active'" if active_only else '') + ' ORDER BY created,id', (owner,))
            return [dict(row) for row in rows]

    def observe(self, owner, source_id, text, candidates=None):
        """Persist only the current authenticated user turn, atomically and once.

        The adapter must not pass assistant output as text. Duplicate transport
        requests don't count as repeated confirmation. A new user event does.
        """
        self._owner(owner)
        if not isinstance(source_id, str) or not source_id or len(source_id) > 220:
            raise ValueError('stable source event ID required')
        if not isinstance(text, str) or len(text) > 12000:
            raise ValueError('invalid current user text')
        candidates = extract_explicit(text) if candidates is None else list(candidates)
        if len(candidates) > 16:
            raise ValueError('too many candidates')
        for candidate in candidates:
            candidate.validate(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        receipt = {'status': 'unchanged', 'saved': [], 'confirmed': [], 'rejected': 0}
        now = self.clock()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT digest FROM processed WHERE owner=? AND source_id=?',
                                  (owner, source_id)).fetchone()
            if previous:
                if previous['digest'] != digest:
                    raise ValueError('source event ID reused with different text')
                return {**receipt, 'status': 'duplicate'}
            for c in candidates:
                score = 0.5*c.confidence + 0.35*c.stability + 0.15*c.importance
                if c.confidence < self.policy.admission or c.stability < 0.7 or (
                        score < self.policy.admission and not c.explicit_keep):
                    receipt['rejected'] += 1
                    continue
                rows = db.execute("SELECT * FROM facts WHERE owner=? AND subject=? "
                    "AND predicate=? AND status='active'", (owner, c.subject, c.predicate)).fetchall()
                prior = next((r for r in rows if r['id'] == c.supersedes), None)
                if c.supersedes and prior is None:
                    raise ValueError('correction must target an active fact of the same owner and subject')
                exact = next((r for r in rows if (r['value'], r['polarity'], r['degree']) ==
                             (c.value, c.polarity, c.degree)), None)
                if exact:
                    if prior and prior['id'] != exact['id']:
                        db.execute("UPDATE facts SET status='superseded' WHERE id=? AND owner=?", (prior['id'], owner))
                    db.execute('UPDATE facts SET confirmed=?,confirmations=confirmations+1,'
                               'pinned=MAX(pinned,?) WHERE id=? AND owner=?',
                               (now, int(c.explicit_keep), exact['id'], owner))
                    receipt['confirmed'].append(exact['id'])
                    continue
                # A name is single-valued. Preferences can coexist; only a direct
                # polarity/degree correction to the same object invalidates one.
                conflicts = [r for r in rows if c.predicate == 'name' or r['value'] == c.value or r['id'] == c.supersedes]
                for row in conflicts:
                    db.execute("UPDATE facts SET status='superseded' WHERE id=? AND owner=?",
                               (row['id'], owner))
                ident = uuid.uuid4().hex
                db.execute('INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ident, owner, c.subject, c.predicate, c.value, c.polarity, c.degree,
                     c.evidence, source_id, now, now, 1, c.confidence, c.importance,
                     c.stability, int(c.explicit_keep), 'active', conflicts[-1]['id'] if conflicts else None))
                receipt['saved'].append(ident)
            db.execute('INSERT INTO processed VALUES (?,?,?)', (owner, source_id, digest))
        if receipt['saved'] or receipt['confirmed']:
            receipt['status'] = 'saved'
        return receipt

    def forget_event(self, owner, source_id, text, ids):
        """Delete selected facts and record the user event in one transaction.

        A transport replay cannot delete facts reintroduced by a later event.
        All targets must belong to this owner; an invalid target aborts the batch.
        """
        self._owner(owner)
        if not isinstance(source_id, str) or not 0 < len(source_id) <= 220:
            raise ValueError('stable source event ID required')
        if not isinstance(text, str) or len(text) > 12000:
            raise ValueError('invalid current user text')
        if not isinstance(ids, list) or len(ids) > 16 or any(not isinstance(i,str) for i in ids):
            raise ValueError('invalid deletion targets')
        digest = hashlib.sha256(text.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT digest FROM processed WHERE owner=? AND source_id=?',
                                  (owner,source_id)).fetchone()
            if previous:
                if previous['digest'] != digest:
                    raise ValueError('source event ID reused with different text')
                return {'status':'duplicate','operation':'delete','count':0}
            targets = []
            for ident in set(ids):
                row = db.execute('SELECT * FROM facts WHERE owner=? AND id=?', (owner,ident)).fetchone()
                if row is None:
                    raise ValueError('deletion target is unavailable to this owner')
                targets.append(row)
            count = 0
            for row in targets:
                count += db.execute('DELETE FROM facts WHERE owner=? AND subject=? AND predicate=? AND value=?',
                    (owner,row['subject'],row['predicate'],row['value'])).rowcount
            db.execute('INSERT INTO processed VALUES (?,?,?)',(owner,source_id,digest))
        return {'status':'deleted' if targets else 'unavailable','operation':'delete','count':count}

    def forget(self, owner, *, subject=None, predicate=None, value=None):
        """Explicit deletion, including superseded versions; no score-based purge.

        Event digests remain to prevent a replay of an old write restoring deleted
        facts. They contain neither source text nor fact values.
        """
        self._owner(owner)
        clauses, args = ['owner=?'], [owner]
        for key, term in [('subject', subject), ('predicate', predicate), ('value', value)]:
            if term is not None:
                if not isinstance(term, str) or not term:
                    raise ValueError('invalid deletion selector')
                clauses.append(key + '=?')
                args.append(term)
        with self.connect() as db:
            count = db.execute('DELETE FROM facts WHERE ' + ' AND '.join(clauses), args).rowcount
        return {'status': 'deleted', 'count': count}

    def rank(self, owner, relevance, *, now=None):
        """Relevance is a callable over this owner's active records only.

        Semantic scoring can reuse a local embedding/reranker. Reject nonfinite
        scores; low relevance cannot be rescued by importance or recency.
        Retrieval never counts as user confirmation or changes retention.
        """
        records = self.records(owner)
        scores = list(relevance(records)) if records else []
        if len(scores) != len(records):
            raise ValueError('relevance result length mismatch')
        now, p, ranked = self.clock() if now is None else now, self.policy, []
        for row, related in zip(records, scores):
            if isinstance(related, bool) or not math.isfinite(related) or not 0 <= related <= 1:
                raise ValueError('invalid relevance score')
            if related < p.minimum_relevance:
                continue
            strength = min(p.max_strength, 1 + p.confirmation_gain * (row['confirmations'] - 1))
            age_days = max(0, now-row['confirmed']) / 86400
            recent = math.exp(-math.log(2) * age_days / (p.half_life_days * strength))
            score = p.relevance_weight*related + p.importance_weight*row['importance'] + p.recency_weight*recent
            ranked.append({**row, 'relevance': related, 'recency': recent, 'score': score})
        ranked.sort(key=lambda r: (-r['score'], -r['confirmed'], r['id']))
        selected = []
        for row in ranked:
            if len(selected) >= p.limit:
                break
            if len(json.dumps(selected + [row], ensure_ascii=False)) <= p.context_chars:
                selected.append(row)
        return selected
