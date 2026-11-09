"""Planner proposal validation; no model loading, transport, or storage ownership."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import re
import sqlite3

from .memory import (Candidate, LifecycleStore, extract_explicit,
    statement_value, statement_subject_tail, statement_correction_matches, statement_topic,
    STATEMENT_ATTRIBUTES, mixed_behavior_instruction, correction_body)


def predicate_object(subject, predicate, evidence):
    """Read the object of a supported explicit clause, preserving its scope.

    This validates a proposed value; it never rewrites the model's candidate.
    Unrecognized clause forms return None, rather than accepting any substring.
    """
    text = statement_value(evidence)
    if subject == 'user' and predicate == 'name' and text.startswith('叫我'):
        return text[len('叫我'):].strip() or None
    who = '我' if subject == 'user' else subject.removeprefix('related:')
    if not text.startswith(who):
        return None
    tail = text[len(who):]
    if predicate == 'preference':
        match = re.fullmatch(r'(?:一直|平时|通常|现在|以后)?'
            r'(最喜欢|不再喜欢|不喜欢|喜欢|最爱|不爱|爱吃|爱喝|爱好|偏爱|讨厌)(.+)', tail)
        if match:
            value = match[2].strip()
            return value.removesuffix('了') if match[1] in {'不再喜欢', '不喜欢', '不爱', '讨厌'} else value
        match = re.fullmatch(r'(?:的(?:喜好|偏好|爱好|兴趣|口味)'
            r'|把(?:自己)?(?:最喜欢|喜欢|不喜欢)[^，,。！!；;\n]{1,30})'
            r'(?:是|为|改成|改为)(.+)', tail)
        return match[1].strip() if match else None
    if predicate == 'name':
        match = re.fullmatch(r'(?:叫|的名字(?:是|叫|改成|改为))(.+)', tail)
        return match[1].strip() if match else None
    if predicate == 'allergy':
        match = re.fullmatch(r'对(.+?)(?:不再过敏|不过敏|过敏)(?:了)?', tail)
        return match[1].strip() if match else None
    return None


def model_candidates(proposals, current):
    """Validate the model's extraction against explicit current-turn evidence.

    This deliberately rejects inferred facts; admission confidence is not a truth
    score. Structured output never gets to select the persistence owner.
    """
    accepted = []
    if re.search(r'[“”「」『』\"？?]|开玩笑|假如|假设|比如|例如|才怪', current):
        return accepted
    for item in proposals[:16]:
        if not isinstance(item, dict):
            continue
        evidence, value = item.get('evidence'), item.get('value')
        if not isinstance(evidence, str) or not isinstance(value, str) or not value or value not in evidence or evidence not in current:
            continue
        if mixed_behavior_instruction(evidence):
            continue
        # Require a whole self-report clause, not a substring of a quoted or
        # attributed claim. Subjects other than the speaker must be explicit.
        clauses = [s.strip() for s in re.split(r'[，,。！!；;\n]+', current)]
        body = correction_body(evidence)
        whole_clause = evidence.rstrip('。！!；;\n ') in clauses
        if body is not None:
            # This exception covers the complete current utterance only. Never
            # strip a reporting frame, another clause, or a trailing qualifier.
            whole_clause = (current.strip().rstrip('。！!；;\n ') == evidence.strip().rstrip('。！!；;\n ')
                            and body in clauses and item.get('predicate') == 'statement')
        if not whole_clause or re.search(r'今天|这次|暂时|临时|刚才|可能|也许|如果|不一定', evidence):
            continue
        subject, predicate = item.get('subject'), item.get('predicate')
        if not isinstance(predicate, str):
            continue
        if predicate == 'statement':
            # A generic fact must retain the entire proposition. Substrings
            # such as "需要拐杖" cannot erase its original subject or negation.
            # The planner may retain terminal punctuation. Canonicalize only
            # that suffix after verifying the original value is verbatim;
            # the full source evidence and all admission scores stay intact.
            value = value.rstrip('。！!；;\n ')
            if (value != statement_value(evidence) or statement_subject_tail(subject, value) is None
                    or item.get('polarity') != 'positive' or item.get('degree') != 'ordinary'
                    or re.search(r'明天|后天|今晚|明晚|这周|下周|周末|一会儿|正在|刚刚|想要|打算|准备|忽略|指令|扮演|改名|你必须|你应该|是不是|有没有|会不会|能不能|需不需要|[吗么呢吧]$', value)):
                continue
            importance, stability = item.get('importance'), item.get('stability')
            if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                   for v in (importance, stability)):
                continue
            supersedes = item.get('supersedes')
            if supersedes is not None and (not isinstance(supersedes, str) or not re.search(
                    r'改成|改为|更正|不再|纠正|不是|没有|不住在|不需要|不使用|不负责|不从事', current)):
                continue
            candidate = Candidate(subject, predicate, value, evidence, importance=importance,
                stability=stability, explicit_keep=bool(re.match(r'请记住|记住|别忘了', evidence)),
                supersedes=supersedes)
            try:
                candidate.validate(current)
            except (TypeError, ValueError):
                continue
            accepted.append(candidate)
            continue
        if re.search(r'不是|说|告诉', evidence):
            continue
        self_report_forms = {
            'preference': r'^(?:我(?:最喜欢|不再喜欢|不喜欢|喜欢|最爱|不爱|爱吃|爱喝|爱好|偏爱|讨厌)'
                          r'|我的(?:喜好|偏好|爱好|兴趣|口味)(?:是|为|改)'
                          r'|我把(?:自己)?(?:最喜欢|喜欢|不喜欢))',
            'name': r'^(?:我叫|我的名字(?:是|叫|改)|叫我)',
            'allergy': r'^我对',
        }
        if predicate not in self_report_forms:
            continue
        if subject == 'user':
            # Accept explicit speaker grammar, rather than trying to enumerate
            # every possible relative/colleague after "我" or "我的".
            statement = re.sub(r'^(?:请记住|记住|别忘了)', '', evidence)
            if not re.match(self_report_forms[predicate], statement):
                continue
        elif isinstance(subject, str) and subject.startswith('related:'):
            if not evidence.startswith(subject[len('related:'):]):
                continue
        else:
            continue
        if predicate_object(subject, predicate, evidence) != value:
            continue
        negative = bool(re.search(r'不喜欢|不爱|讨厌|不再喜欢|不过敏|不再过敏', evidence))
        if item.get('polarity', 'positive') != ('negative' if negative else 'positive'):
            continue
        favorite = bool(re.search(r'最喜欢|最爱', evidence))
        if item.get('degree', 'ordinary') != ('favorite' if favorite else 'ordinary'):
            continue
        if predicate == 'preference' and not re.search(r'喜欢|爱吃|爱喝|爱好|讨厌|偏爱|最爱', evidence):
            continue
        if predicate == 'allergy' and '过敏' not in evidence:
            continue
        supersedes = item.get('supersedes')
        if supersedes is not None and (not isinstance(supersedes, str) or not re.search(r'改成|改为|更正|不再|纠正', current)):
            continue
        c = Candidate(subject, predicate, value, evidence,
                      polarity=item.get('polarity', 'positive'), degree=item.get('degree', 'ordinary'),
                      importance=1.0 if predicate == 'allergy' else 0.5,
                      explicit_keep=bool(re.match(r'请记住|记住|别忘了', evidence)), supersedes=supersedes)
        try:
            c.validate(current)
        except (TypeError, ValueError):
            continue
        accepted.append(c)
    return accepted


def deletion_matches(record, current):
    """Ground a proposed destructive target in the user's explicit scope.

    Valid owner-scoped IDs alone do not establish intent. Accept complete facts
    and explicit subject/attribute forms; ambiguous pronouns and unsupported
    paraphrases leave the record intact. The model never expands this scope.
    """
    match = re.fullmatch(r'(?:请)?(?:忘记|删掉|删除|不要再记住)(.+?)[。！!\s]*', current.strip())
    if not match:
        return False
    target = match[1]
    if re.search(r'[，,；;？?“”「」『』\"\n]|但是|除了|除外|保留|不要|别删|如果|假如', target):
        return False
    target = re.sub(r'(?:这件事|这个事实|这条记录|的(?:记录|信息|记忆))$', '', target)
    subject = '我' if record['subject'] == 'user' else record['subject'].removeprefix('related:')
    if not target.startswith(subject):
        return False
    tail = target[len(subject):]
    predicate = record['predicate']
    if predicate == 'statement':
        if target == record['value']:
            return True
        if tail not in {'的' + attribute for attribute in STATEMENT_ATTRIBUTES}:
            return False
        topic = statement_topic(record['subject'], record['value'])
        # Reuse the same attribute aliases as correction matching.
        requested = statement_topic(record['subject'], subject + tail + '是待删除值')
        return topic is not None and topic == requested
    if predicate == 'name':
        return tail in {'的名字', '的姓名'} or target == statement_value(record['evidence'])
    if predicate == 'preference':
        return (tail in {'的喜好', '的偏好', '对' + record['value'] + '的喜好',
                         '对' + record['value'] + '的偏好'}
                or target == statement_value(record['evidence']))
    if predicate == 'allergy':
        return tail in {'的过敏', '的过敏情况', '对' + record['value'] + '过敏',
                        '对' + record['value'] + '的过敏'}
    return False


def explicit_attributes(current, records):
    """Propose a complete named attribute without asking a model to copy it.

    This narrow syntax path does not classify the surrounding dialogue frame.
    Runtime still gates it with the same quotation/joke/temporary safeguards as
    other explicit candidates. Unknown subjects and compound clauses abstain.
    """
    evidence = current.strip()
    value = statement_value(evidence)
    attribute = '(?:' + '|'.join(STATEMENT_ATTRIBUTES) + ')'
    if not re.fullmatch(r'我的' + attribute + r'(?:不是|是|为|不在|在|改成|改为)'
                        r'[^，,。！!；;：:\n]+', value):
        return []
    if re.search(r'也许|可能|大概|似乎|应该|不一定|据说|听说|好像|'
                 r'如果|假如|暂时|临时|这次|今天|明天|后天|今晚|明晚|'
                 r'这周|下周|本周|今年|明年|刚才|正在|打算|准备|想要|'
                 r'而是|但是|不过|并且', value):
        return []
    proposals = [{'subject': 'user', 'predicate': 'statement', 'value': value,
                  'evidence': evidence,
                  'polarity': 'positive', 'degree': 'ordinary',
                  'importance': 0.7, 'stability': 1.0}]
    correction = correction_body(evidence) is not None or re.match(
        r'^我的' + attribute + r'(?:改成|改为)', value)
    if correction:
        topic = statement_topic('user', value)
        matches = [row for row in records if row['subject'] == 'user'
                   and row['predicate'] == 'statement'
                   and statement_topic('user', row['value']) == topic]
        if len(matches) > 1:
            return []
        if len(matches) == 1 and matches[0]['value'] != value:
            proposals[0]['supersedes'] = matches[0]['id']
    return model_candidates(proposals, current)


def answer_operations(operations):
    """Keep execution outcomes, without random persistence IDs in model input."""
    result = []
    for operation in operations:
        item = {k: v for k, v in operation.items() if k not in {'saved', 'confirmed'}}
        for field in ('saved', 'confirmed'):
            if field in operation:
                item[field + '_count'] = len(operation[field])
        result.append(item)
    return result


def explicit_quotation_frame(messages):
    """Keep an explicitly announced quotation from authorizing memory writes.

    This is a narrow negative guard, not a general dialogue classifier. Only
    supplied user history can open it. A standalone explicit end closes it;
    sentence counts and an inferred topic change cannot authorize new writes.
    No state is inferred from assistant text or from omitted history.
    """
    opening = re.compile(
        r'^(?:(?:接下来|下面|以下|随后)(?:的)?(?:[一二两三四五六七八九十0-9]+句)?'
        r'|下[一二两三四五六七八九十0-9]+句)'
        r'(?:是|我(?:会|要|来)?(?:读|念|引用|转述)|引用|转述|朗读)'
        r'[^，,。！!；;：:\n？?“”「」『』\"]{0,40}'
        r'(?:台词|对白|引文|原文|转述内容)[。！!\s]*$')
    closing = re.compile(r'^(?:(?:引用|台词|对白|引文|转述)(?:结束|到此结束)'
                         r'|结束(?:引用|转述)|我(?:已经)?读完了)[。！!\s]*$')
    active = None
    for index, message in enumerate(messages):
        if message['role'] != 'user':
            continue
        text = message['content'].strip()
        if closing.fullmatch(text):
            active = None
        elif index < len(messages) - 1 and opening.fullmatch(text) and not re.search(
                r'不是|并非|而非|不要|不会|不再|假如|如果', text):
            active = {'kind': 'quotation', 'source_message_index': index,
                      'evidence': text, 'scope': 'explicit_user_history_until_end'}
    return active
