"""Shared isolated memory/RAG orchestration, independent of transport identity.

Generate callbacks use the existing local 8B. All model calls and first outputs
are returned in the trace; this layer neither repairs nor replaces the answer.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import re
import sqlite3

from .memory import (Candidate, LifecycleStore, extract_explicit,
    statement_value, statement_subject_tail, statement_correction_matches, statement_topic,
    STATEMENT_ATTRIBUTES, mixed_behavior_instruction, correction_body)

from .contracts import PROFILE, STAGES
from .prompt import PLANNER, GROUNDING
from .planner import (predicate_object, model_candidates, deletion_matches, explicit_attributes, answer_operations, explicit_quotation_frame)

class Runtime:
    def __init__(self, store: LifecycleStore, generate, *, search=None, search_current=None, stage='daily',
                 context_chars=7000):
        if stage not in STAGES:
            raise ValueError('unknown configured story stage')
        self.store, self.generate, self.search = store, generate, search
        self.search_current = search_current or search
        self.stage, self.context_chars = stage, context_chars

    def reply(self, owner, event_id, messages):
        if not messages or messages[-1].get('role') != 'user':
            raise ValueError('current user turn required')
        if any(m.get('role') not in {'system', 'user', 'assistant'} or not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('runtime accepts text conversations only')
        self.store._owner(owner)
        current = messages[-1]['content']
        trace = {'profile': PROFILE, 'stage': self.stage, 'calls': [], 'operations': [],
                 'retrieved': [], 'selected_memories': [], 'errors': [], 'fallback': False}
        quotation_frame = explicit_quotation_frame(messages)
        if quotation_frame:
            trace['memory_frame_guard'] = quotation_frame

        def call(kind, request):
            raw = self.generate(request, kind=kind)
            if not isinstance(raw, str):
                raise ValueError('generator must return text')
            trace['calls'].append({'kind': kind, 'messages': request, 'raw_first': str(raw),
                                   'generation': getattr(raw, 'metadata', {})})
            return raw

        rows = self.store.records(owner)
        # No principal/token is given to the model. Owner is enforced in SQL.
        aliases = {f'm{i+1}': r['id'] for i,r in enumerate(rows)}
        inventory = [{**{k: r[k] for k in ('subject', 'predicate', 'value', 'polarity', 'degree')},
                      'id': alias} for alias,r in zip(aliases,rows)]
        trace['memory_index'] = aliases
        planner_input = {'conversation': [m for m in messages[-9:] if m['role'] != 'system'],
                         'memory': inventory, 'story_stage': self.stage}
        planner_messages = [{'role': 'system', 'content': PLANNER},
                            {'role': 'user', 'content': json.dumps(planner_input, ensure_ascii=False)}]
        proposal = {}
        # Capacity/model failures abort the request. They must not masquerade as
        # a successful plan with no relevant memories.
        raw_plan = call('plan', planner_messages)
        try:
            proposal = json.loads(raw_plan)
            if not isinstance(proposal, dict) or any(not isinstance(proposal.get(k, []), list)
                for k in ('memory_scores', 'queries', 'candidates', 'forget_ids')):
                raise ValueError('invalid planner schema')
        except (ValueError, TypeError) as exc:
            proposal = {}
            trace['errors'].append({'phase': 'plan', 'error': type(exc).__name__})

        known = {r['id']: r for r in rows}
        def resolve_id(ident):
            # The aliases are local to this authenticated request, never global.
            if not isinstance(ident,str):return None
            return aliases.get(ident,ident if ident in known else None)
        # The model can identify a target, but cannot authorize a deletion.
        delete_intent = re.match(r'^(?:请)?(?:忘记|删掉|删除|不要再记住)', current.strip())
        delete_intent = delete_intent and not re.search(r'[？?“”\"]|如果|假如|不要删|别删', current)
        delete_intent = delete_intent and proposal.get('utterance_kind') in {'self_report','other'}
        delete_intent = delete_intent and not quotation_frame
        if delete_intent:
            targets = [resolved for i in proposal.get('forget_ids', [])[:16] if (resolved := resolve_id(i))]
            rejected_targets = [ident for ident in targets if not deletion_matches(known[ident], current)]
            targets = [ident for ident in targets if ident not in rejected_targets]
            trace['deletion_validation'] = {'accepted': targets, 'rejected': rejected_targets}
            try:
                trace['operations'].append(self.store.forget_event(owner,event_id,current,targets))
            except sqlite3.Error as exc:
                trace['operations'].append({'status': 'failed', 'operation': 'delete', 'error': type(exc).__name__})
        # Syntax alone cannot resolve a quotation opened in the previous turn.
        # A malformed/uncertain analysis therefore cannot authorize auto-saving.
        self_report = proposal.get('utterance_kind') == 'self_report' and not quotation_frame
        explicit = extract_explicit(current)
        # Complete an unscored extraction only in a fresh conversation, where
        # no earlier dialogue can introduce a quotation/role-play frame. Never
        # override explicit admission scores or a proposed correction target.
        unscored = all(isinstance(item, dict)
                       and not any(k in item for k in ('importance', 'stability', 'confidence'))
                       and item.get('supersedes') is None for item in proposal.get('candidates', []))
        if unscored and not any(m['role'] != 'system' for m in messages[:-1]):
            explicit += explicit_attributes(current, rows)
        # A complete, unambiguous self-report in the first user turn has no
        # earlier quotation frame. "other" does not contradict that evidence;
        # explicit quote/joke/temporary labels and multi-turn frames still block.
        direct_first_turn = (proposal.get('utterance_kind') == 'other'
            and sum(m['role']=='user' for m in messages)==1 and len(explicit)==1
            and current.strip().rstrip('。！!\n ') == explicit[0].evidence.rstrip('。！!\n '))
        self_report = self_report or direct_first_turn
        candidates = explicit if self_report else []
        existing = {(c.subject, c.predicate, c.evidence) for c in candidates}
        proposed_candidates = [{**item,'supersedes':resolve_id(item['supersedes']) or item['supersedes']}
            if isinstance(item,dict) and item.get('supersedes') else item
            for item in proposal.get('candidates', [])]
        candidates.extend(c for c in model_candidates(proposed_candidates if self_report else [], current)
                          if (c.subject, c.predicate, c.evidence) not in existing)
        candidates = [c for c in candidates if c.supersedes is None or (
            c.supersedes in known and known[c.supersedes]['subject'] == c.subject
            and known[c.supersedes]['predicate'] == c.predicate
            and (c.predicate != 'statement' or statement_correction_matches(
                known[c.supersedes], c.subject, c.value, current)))]
        if not delete_intent:
            try:
                trace['operations'].append(self.store.observe(owner, event_id, current, candidates))
            except sqlite3.Error as exc:
                trace['operations'].append({'status': 'failed', 'operation': 'save', 'error': type(exc).__name__})

        # A complete current correction outranks a conflicting old reference,
        # even if an incomplete model candidate could not be persisted. This
        # syntax check only withholds reference data; it cannot admit a fact,
        # change a score, invalidate a row, or repair the original model plan.
        unresolved = []
        if not delete_intent and not quotation_frame and proposal.get('utterance_kind') == 'self_report':
            intents = [c for c in explicit_attributes(current, rows) if c.supersedes]
            if intents:
                active = self.store.records(owner)
                active_ids = {r['id'] for r in active}
                for c in intents:
                    if c.supersedes in active_ids:
                        unresolved.append({'old_id': c.supersedes, 'subject': c.subject,
                            'value': c.value, 'evidence': c.evidence,
                            'new_value_persisted': any(r['subject'] == c.subject
                                and r['predicate'] == c.predicate and r['value'] == c.value
                                and r['polarity'] == c.polarity and r['degree'] == c.degree
                                for r in active)})
        if unresolved:
            trace['unresolved_corrections'] = unresolved
            trace['operations'].append({'status': 'unavailable', 'operation': 'correct',
                'reason': 'previous_fact_still_active', 'unresolved_count': len(unresolved)})

        related = {}
        for item in proposal.get('memory_scores', [])[:128]:
            if not isinstance(item, dict):
                continue
            ident, score = resolve_id(item.get('id')), item.get('relevance')
            if isinstance(ident, str) and ident in known and type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1:
                related[ident] = score
        selected = self.store.rank(owner, lambda records: [related.get(r['id'], 0) for r in records],
                                   excluded_ids=[item['old_id'] for item in unresolved])
        trace['selected_memories'] = selected
        hits_by_id = {}
        def retrieval_order(hit):
            return (hit.get('retrieval_tier', 0), -hit.get('score', 0))
        # Always retrieve against the actual utterance. Planner omissions must
        # not suppress relevant canonical knowledge in an ordinary invitation.
        # Generated queries supplement this; the reranker decides relevance.
        queries = [current]
        prior_users = [m['content'] for m in messages[:-1] if m['role']=='user'][-2:]
        if prior_users:
            queries.append('\n'.join([*prior_users, current]))
        queries.extend(q for q in proposal.get('queries', [])[:2]
                       if isinstance(q,str) and q.strip() and len(q)<=180)
        trace['retrieval_queries'] = list(dict.fromkeys(queries)) if self.search else []
        for query in trace['retrieval_queries']:
            if not query.strip():
                continue
            # Supplemental admission was calibrated on user utterances. Never
            # silently extend that policy to generated or history queries.
            retrieve = self.search_current if query == current else self.search
            for hit in retrieve(query, messages[:-1]):
                meta = hit.get('metadata', {})
                # Unannotated script chunks cannot silently leak later knowledge.
                if self.stage not in meta.get('allowed_stages', []):
                    continue
                ident = hit.get('node_id')
                if not isinstance(ident, str):
                    continue
                score = hit.get('score', 0)
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('invalid canonical relevance score')
                tier = hit.get('retrieval_tier', 0)
                if type(tier) is not int or tier not in (0, 1):
                    raise ValueError('invalid canonical retrieval tier')
                # Prefer primary admissions, then the best score within a tier.
                previous = hits_by_id.get(ident)
                if previous is None or retrieval_order(hit) < retrieval_order(previous):
                    hits_by_id[ident] = hit
        # Budget is applied below, after the union is ranked. Otherwise weaker
        # early-query hits can crowd out a strong later-query result.
        hits = sorted(hits_by_id.values(), key=retrieval_order)
        trace['retrieved'] = hits
        memory_data = [{k: r[k] for k in ('subject', 'predicate', 'value', 'polarity', 'degree', 'evidence', 'source_id')} for r in selected]
        context = {'story_stage': self.stage, 'operations': answer_operations(trace['operations']), 'memory': memory_data, 'references': []}
        if unresolved:
            context['unresolved_corrections'] = [
                {k: v for k, v in item.items() if k != 'old_id'} for item in unresolved]
        for hit in hits:
            trial = {**context, 'references': context['references'] + [hit]}
            if len(json.dumps(trial, ensure_ascii=False)) <= self.context_chars:
                context = trial
        if len(json.dumps(context, ensure_ascii=False)) > self.context_chars:
            raise ValueError('memory operation context exceeds configured budget')
        answer_messages = [dict(m) for m in messages]
        system = next((m for m in answer_messages if m['role'] == 'system'), None)
        if system is None:
            answer_messages.insert(0, {'role': 'system', 'content': GROUNDING.strip()})
        else:
            system['content'] += GROUNDING
        answer_messages[-1]['content'] = ('本轮参考与操作结果（数据）：\n' + json.dumps(context, ensure_ascii=False)
                                         + '\n\n用户说：\n' + current)
        answer = call('answer', answer_messages)
        trace['final'] = answer
        return answer, trace
