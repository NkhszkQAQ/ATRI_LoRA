"""Owner-scoped memory lifecycle. No model, network, or production DB defaults.

Extraction, admission, ranking, invalidation, and deletion are separate operations.
The caller supplies an authenticated owner; candidate data cannot choose an owner.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
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


# These boundaries identify the speaker/entity of a complete assertion; they do
# not claim to decide whether arbitrary prose is true or durably relevant.
STATEMENT_ATTRIBUTES = ('生日', '出生日期', '职业', '工作', '学校', '专业', '母语',
                        '家乡', '居住地', '住址', '现住址', '家庭', '家庭成员',
                        '身高', '血型', '作息', '长期目标')
_ATTRIBUTE_FORM = re.compile(r'^的(?:' + '|'.join(STATEMENT_ATTRIBUTES) +
                             r')(?:是|不是|为|在|不在|有|需要|改成|改为).+')
_ASSERTION_FORM = re.compile(
    r'^(?:长期|一直|平时|通常|从小|现在|已经|不再)?'
    r'(?:不是|是|没有|有(?!点)|不住在|住在|来自|患有|就读于|毕业于|不从事|从事|不负责|负责|不使用|使用|不需要|需要|不能|不会|必须|只能|养了|养着|养有|养).+')


def correction_body(evidence):
    """Recognize a standalone correction cue followed by one complete clause."""
    match = re.fullmatch(r'(?:更正|纠正)(?:一下)?[，,]\s*([^，,。！!；;\n]+)[。！!；;\n ]*', evidence.strip())
    return match[1].strip() if match else None


def statement_value(evidence):
    """Keep the whole assertion, including its subject and any negation."""
    text = correction_body(evidence) or evidence.strip()
    return re.sub(r'^(?:请记住|记住|别忘了)', '', text).rstrip('。！!；;\n ')


def mixed_behavior_instruction(text):
    """Do not admit a fact clause joined to an instruction changing controls."""
    return bool(re.search(r'(?:并(?:且)?|然后|同时|还要|而且)(?:请)?'
                          r'(?:忽略|关闭|跳过|绕过|改写|更改|删除)'
                          r'[^，,。！!；;\n]*(?:权限|验证|规则|指令|系统提示)', text))


def statement_subject_tail(subject, value):
    """Return the assertion after a verbatim subject, or None if unresolved."""
    if subject == 'user':
        if not value.startswith('我'):
            return None
        tail = value[1:]
    elif isinstance(subject, str) and subject.startswith('related:'):
        source = subject[len('related:'):]
        if (len(source) < 2 or source in {'我的', '你的', '我们', '你们', '他们', '她们', '有人'}
                or source in {'我的' + attr for attr in STATEMENT_ATTRIBUTES}
                or not value.startswith(source)):
            return None
        tail = value[len(source):]
    else:
        return None
    # A reporting verb governing another assertion is not an identity/possession
    # fact. Match the clause structure, not a character inside 小说/说明书/说书人.
    plain_tail = re.sub(r'^(?:长期|一直|平时|通常|从小|现在|已经)', '', tail)
    if re.match(r'^(?:不是|是|没有|有|不再)?(?:在)?告诉', plain_tail) or re.match(
            r'^(?:不是|是|没有|有|不再)?(?:在)?说(?:过|道)?'
            r'(?:[我你他她它这那，,:：]|自己|别人|有人|.{0,80}(?:是|喜欢|需要|应该|可以|不能|没有))', plain_tail):
        return None
    if _ATTRIBUTE_FORM.match(tail) or _ASSERTION_FORM.match(tail) or re.fullmatch(
            r'(?:长期|一直|现在|不再)?(?:不在|在).{1,80}(?:工作|读书|上学|就读)', tail):
        return tail
    return None


def statement_topic(subject, value):
    """A few explicit attribute frames support scoped correction comparisons.

    Unknown topics return None instead of allowing one generic statement to
    overwrite any other. This is not a model-generated category or database key.
    """
    tail = statement_subject_tail(subject, value)
    if tail is None:
        return None
    match = re.match(r'^的(' + '|'.join(STATEMENT_ATTRIBUTES) + r')(?:是|不是|为|在|不在|有|需要|改成|改为)', tail)
    if match:
        return {'出生日期': '生日', '现住址': '居住地', '住址': '居住地', '工作': '职业'}.get(match[1], match[1])
    tail = re.sub(r'^(?:长期|一直|平时|通常|从小|现在|已经|不再)', '', tail)
    for prefix, topic in [('不住在', '居住地'), ('住在', '居住地'), ('就读于', '学校'), ('从事', '职业')]:
        if tail.startswith(prefix):
            return topic
    return None


def statement_correction_matches(old, subject, value, source):
    before, after = statement_topic(subject, old['value']), statement_topic(subject, value)
    return ((before is not None and before == after) or old['value'] in source
            or statements_contradict(subject, old['value'], value))


def statements_contradict(subject, before, after):
    """Recognize a literal positive/negative pair, not semantic incompatibility.

    Different occupations, addresses, or family facts can coexist unless the
    user explicitly corrects one. Never infer contradictions from similarity.
    """
    def proposition(value):
        tail = statement_subject_tail(subject, value)
        if tail is None:
            return None
        tail = re.sub(r'^(?:长期|一直|平时|通常|从小|现在|已经)', '', tail)
        no_longer = tail.startswith('不再')
        if no_longer:
            tail = tail[len('不再'):]
        attribute = re.match(r'^的(?:' + '|'.join(STATEMENT_ATTRIBUTES) + ')', tail)
        prefix = attribute[0] if attribute else ''
        rest = tail[len(prefix):]
        pairs = [('不是', '是'), ('没有', '有'), ('不住在', '住在'), ('不需要', '需要'),
                 ('不使用', '使用'), ('不负责', '负责'), ('不从事', '从事'), ('不在', '在')]
        for negative, positive in pairs:
            if rest.startswith(negative):
                return prefix + positive + rest[len(negative):], no_longer
            if rest.startswith(positive):
                return prefix + rest, not no_longer
        return None
    old, new = proposition(before), proposition(after)
    return old is not None and new is not None and old[0] == new[0] and old[1] != new[1]


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
                not self.subject.startswith('related:') or not self.subject.removeprefix('related:').strip()
                or self.subject.removeprefix('related:') in {
                    '我', '我的', '自己', '你', '你的', '他', '她', '他们', '她们', '我们', '你们'})):
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
        if self.predicate == 'statement' and (
                self.value != statement_value(self.evidence)
                or statement_subject_tail(self.subject, self.value) is None
                or self.polarity != 'positive' or self.degree != 'ordinary'):
            raise ValueError('statement must preserve its complete attributed assertion')


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
        if mixed_behavior_instruction(clause):
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
        if re.search(r'说|告诉|觉得|可能|也许|应该|不一定|但是|不过(?!敏)|而是|并非|不是|才怪', value):
            continue
        subject = 'user' if who == '我' else 'related:' + who
        predicate, importance, degree = 'preference', 0.5, 'ordinary'
        negative = op in {'不再喜欢', '不喜欢', '讨厌'}
        if op == '叫':
            predicate, importance = 'name', 0.7
        elif op == '对':
            value = value.removesuffix('了')
            if not value.endswith('过敏'):
                continue
            negative = value.endswith(('不过敏', '不再过敏'))
            suffix = '不再过敏' if value.endswith('不再过敏') else '不过敏' if negative else '过敏'
            predicate, value, importance = 'allergy', value[:-len(suffix)].strip(), 1.0
        elif op == '最喜欢':
            degree = 'favorite'
        if not value:
            continue
        # Do not guess a food/drink taxonomy. Preserve the actual reported value.
        candidates.append(Candidate(subject, predicate, value, clause,
            polarity='negative' if negative else 'positive',
            degree=degree, importance=importance, explicit_keep=bool(match['keep'])))
    return candidates


def correction_is_explicit(old, candidate, source):
    """Recognize a current correction independently of its retention scores.

    Contextual self-report classification remains the runtime's responsibility.
    This check cannot authorize another owner's record, infer an omitted fact,
    or turn an arbitrary model-provided supersedes ID into a user instruction.
    """
    if old['subject'] != candidate.subject or old['predicate'] != candidate.predicate:
        return False
    if re.search(r'[“”「」『』\"？?]|开玩笑|逗你|假如|假设|如果|比如|例如|才怪|台词|'
                 r'可能|也许|不一定|暂时|临时|今天|明天|后天|这次|这周|下周|今晚|明晚|'
                 r'刚才|一会儿|正在|打算|准备|忽略|指令|'
                 r'(?:不要|别)(?:更正|纠正|修改|改)', source):
        return False
    clause = candidate.evidence.strip().rstrip('。！!；;\n ')
    body = correction_body(candidate.evidence)
    if body is not None:
        if candidate.predicate != 'statement' or candidate.evidence.strip() != source.strip():
            return False
        clause = body
    if clause not in [s.strip() for s in re.split(r'[，,。！!；;\n]+', source)]:
        return False
    prefix = source[:source.index(candidate.evidence)].strip()
    preceding_cue = body is not None or re.fullmatch(r'(?:请)?(?:更正|纠正)(?:一下)?[，,:：\s]*', prefix)
    if prefix and not preceding_cue:
        return False
    if candidate.predicate == 'statement':
        if statements_contradict(candidate.subject, old['value'], candidate.value):
            return True
        if candidate.supersedes != old['id']:
            return False
        if not statement_correction_matches(old, candidate.subject, candidate.value, source):
            return False
        # A standalone correction cue applies to this evidence. A preceding
        # attribution or a cue about another clause cannot grant this priority.
        tail = statement_subject_tail(candidate.subject, candidate.value)
        in_clause_change = tail is not None and re.match(
            r'^的(?:' + '|'.join(STATEMENT_ATTRIBUTES) + r')(?:改成|改为)', tail)
        return bool(preceding_cue or in_clause_change)
    # These literal forms already distinguish ordinary/favorite, positive/
    # negative, and related-person subjects. Different preferences still coexist.
    fields = ('subject', 'predicate', 'value', 'polarity', 'degree')
    if not any(all(getattr(c, key) == getattr(candidate, key) for key in fields)
               for c in extract_explicit(source)):
        return False
    return ((candidate.predicate == 'name' and old['value'] != candidate.value)
            or (old['value'] == candidate.value and (
                old['polarity'] != candidate.polarity or old['degree'] != candidate.degree)))


class LifecycleStore:
    def __init__(self, path, policy=None, *, clock=time.time):
        self.path, self.policy, self.clock = str(path), policy or Policy(), clock
        Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        with self.connect() as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if tables - {'facts', 'processed', 'fact_corrections'}:
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
                CREATE TABLE IF NOT EXISTS fact_corrections (
                    owner TEXT NOT NULL, old_id TEXT NOT NULL, new_id TEXT NOT NULL,
                    source_id TEXT NOT NULL, PRIMARY KEY(owner,old_id,new_id));
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
            observed_records = set()
            applied_corrections = set()
            for c in candidates:
                rows = db.execute("SELECT * FROM facts WHERE owner=? AND subject=? "
                    "AND predicate=? AND status='active'", (owner, c.subject, c.predicate)).fetchall()
                prior = next((r for r in rows if r['id'] == c.supersedes), None)
                exact = next((r for r in rows if (r['value'], r['polarity'], r['degree']) ==
                             (c.value, c.polarity, c.degree)), None)
                already_corrected = exact is not None and (c.supersedes, exact['id']) in applied_corrections
                targets = ([prior] if prior else []) if c.supersedes else rows
                corrects_existing = already_corrected or any(correction_is_explicit(r, c, text) for r in targets)
                score = 0.5*c.confidence + 0.35*c.stability + 0.15*c.importance
                if c.confidence < self.policy.admission or (not corrects_existing and (
                        c.stability < 0.7 or (score < self.policy.admission and not c.explicit_keep))):
                    receipt['rejected'] += 1
                    continue
                if c.supersedes and prior is None and not already_corrected:
                    raise ValueError('correction must target an active fact of the same owner and subject')
                if c.supersedes and not already_corrected and c.predicate == 'statement' and not statement_correction_matches(
                        prior, c.subject, c.value, text):
                    raise ValueError('statement correction must identify the same attribute or original assertion')
                if exact:
                    if prior and prior['id'] != exact['id']:
                        db.execute("UPDATE facts SET status='superseded' WHERE id=? AND owner=?", (prior['id'], owner))
                        db.execute('INSERT OR IGNORE INTO fact_corrections VALUES (?,?,?,?)',
                                   (owner, prior['id'], exact['id'], source_id))
                    if c.supersedes:
                        applied_corrections.add((c.supersedes, exact['id']))
                    if exact['id'] in observed_records:
                        # Repeated clauses or overlapping extraction paths in
                        # one user event are not independent confirmations.
                        # A later explicit keep still applies to that record.
                        db.execute('UPDATE facts SET pinned=MAX(pinned,?) WHERE id=? AND owner=?',
                                   (int(c.explicit_keep), exact['id'], owner))
                        continue
                    db.execute('UPDATE facts SET confirmed=?,confirmations=confirmations+1,'
                               'pinned=MAX(pinned,?) WHERE id=? AND owner=?',
                               (now, int(c.explicit_keep), exact['id'], owner))
                    receipt['confirmed'].append(exact['id'])
                    observed_records.add(exact['id'])
                    continue
                # A name is single-valued. Preferences can coexist; only a direct
                # polarity/degree correction to the same object invalidates one.
                conflicts = [r for r in rows if c.predicate == 'name' or r['value'] == c.value
                    or r['id'] == c.supersedes or (c.predicate == 'statement'
                        and statements_contradict(c.subject, r['value'], c.value))]
                for row in conflicts:
                    db.execute("UPDATE facts SET status='superseded' WHERE id=? AND owner=?",
                               (row['id'], owner))
                ident = uuid.uuid4().hex
                db.execute('INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ident, owner, c.subject, c.predicate, c.value, c.polarity, c.degree,
                     c.evidence, source_id, now, now, 1, c.confidence, c.importance,
                     c.stability, int(c.explicit_keep), 'active',
                     c.supersedes or (conflicts[-1]['id'] if conflicts else None)))
                for row in conflicts:
                    db.execute('INSERT OR IGNORE INTO fact_corrections VALUES (?,?,?,?)',
                               (owner, row['id'], ident, source_id))
                    # A subsequent model candidate may explicitly identify the
                    # correction already performed by this event's grammar path.
                    applied_corrections.add((row['id'], ident))
                receipt['saved'].append(ident)
                observed_records.add(ident)
                if c.supersedes:
                    applied_corrections.add((c.supersedes, ident))
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
            count = self._delete_families(db, owner, [row['id'] for row in targets])
            db.execute('INSERT INTO processed VALUES (?,?,?)',(owner,source_id,digest))
        return {'status':'deleted' if targets else 'unavailable','operation':'delete','count':count}

    @staticmethod
    def _delete_families(db, owner, targets):
        """Remove same-value versions and cross-value correction history.

        Call inside the caller's write transaction. Traverse both directions:
        an old selected ID must not leave its corrected successor behind. Links
        never cross the authenticated owner, subject, or predicate boundary.
        """
        rows = {row['id']: row for row in db.execute(
            'SELECT id,subject,predicate,value,supersedes FROM facts WHERE owner=?', (owner,))}
        same_value, linked = defaultdict(list), defaultdict(list)
        for row in rows.values():
            same_value[(row['subject'], row['predicate'], row['value'])].append(row['id'])
            prior = rows.get(row['supersedes'])
            if prior and (prior['subject'], prior['predicate']) == (row['subject'], row['predicate']):
                linked[row['id']].append(prior['id'])
                linked[prior['id']].append(row['id'])
        # Replacing an old fact with an already-existing fact can join two
        # histories. One legacy supersedes pointer cannot represent both.
        for edge in db.execute('SELECT old_id,new_id FROM fact_corrections WHERE owner=?', (owner,)):
            old, new = rows.get(edge['old_id']), rows.get(edge['new_id'])
            if old and new and (old['subject'], old['predicate']) == (new['subject'], new['predicate']):
                linked[old['id']].append(new['id'])
                linked[new['id']].append(old['id'])
        pending, selected, visited_groups = list(targets), set(), set()
        while pending:
            ident = pending.pop()
            if ident in selected or ident not in rows:
                continue
            selected.add(ident)
            row = rows[ident]
            group = (row['subject'], row['predicate'], row['value'])
            if group not in visited_groups:
                visited_groups.add(group)
                pending.extend(same_value[group])
            pending.extend(linked[ident])
        if not selected:
            return 0
        count = db.executemany('DELETE FROM facts WHERE owner=? AND id=?',
                               [(owner, ident) for ident in selected]).rowcount
        db.executemany('DELETE FROM fact_corrections WHERE owner=? AND (old_id=? OR new_id=?)',
                       [(owner, ident, ident) for ident in selected])
        return count

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
            db.execute('BEGIN IMMEDIATE')
            targets = [row['id'] for row in db.execute(
                'SELECT id FROM facts WHERE ' + ' AND '.join(clauses), args)]
            count = self._delete_families(db, owner, targets)
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
