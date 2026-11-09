"""Replay archived calls or run fresh model calls through bounded RAG planning.

Replay uses supplied reviewer fixtures to test the program, not the model's
ability to plan. Real mode never reads those fixtures or archived answers.
All outputs are private, new-directory artifacts. No training or deployment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

from _atri_core_bootstrap import core_module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['replay', 'real'], required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--script-source', type=Path, required=True)
    parser.add_argument('--reranker', required=True, help='local frozen reranker snapshot')
    parser.add_argument('--review-fixtures', type=Path)
    parser.add_argument('--lookup-fixtures', type=Path,
                        help='explicit retrieval-needs fixtures for program replay only')
    parser.add_argument('--historical-dir', type=Path)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--base-manifest', type=Path)
    parser.add_argument('--adapter-sha')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--execute', action='store_true', help='required before loading real 8B weights')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must be a new private directory')
    if args.mode == 'replay' and (not args.review_fixtures or not args.lookup_fixtures or not args.historical_dir):
        parser.error('replay needs --lookup-fixtures, --review-fixtures and --historical-dir')
    if args.mode == 'real' and (args.review_fixtures or args.lookup_fixtures or args.historical_dir):
        parser.error('real mode cannot access archived answers or reviewer fixtures')
    if args.mode == 'real' and (not args.execute or not all((args.base, args.adapter,
            args.base_manifest, args.adapter_sha))):
        parser.error('real mode requires --execute and explicit verified model paths')
    root = Path(__file__).resolve().parents[1]
    cases = [json.loads(s) for s in args.cases.read_text().splitlines() if s.strip()]
    if not cases or len({r['id'] for r in cases}) != len(cases):
        parser.error('nonempty cases with distinct IDs required')
    # Freeze cases and program inputs before any result is inspected.
    files = [args.cases, *sorted((root/'src/atri_core').glob('*.py')),
             Path(__file__), *[root/'rag'/n for n in
                ('character_facts.json', 'story_stages.json', 'story_daily_extended.json')]]
    seal = {str(p): digest(p) for p in files}
    historical, fixtures, lookup_fixtures = {}, {}, {}
    if args.mode == 'replay':
        fixtures = json.loads(args.review_fixtures.read_text())
        lookup_fixtures = json.loads(args.lookup_fixtures.read_text())
        seal[str(args.review_fixtures)] = digest(args.review_fixtures)
        seal[str(args.lookup_fixtures)] = digest(args.lookup_fixtures)
        for line in (args.historical_dir/'http-results.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row.get('arm') == 'improved' and row.get('turn') == 0:
                historical[(row['case'], row['seed'])] = row
    generate_real, identity = None, {'model_calls': 0, 'model_loaded': False}
    if args.mode == 'real':
        import subprocess
        active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                check=True, capture_output=True, text=True).stdout.strip()
        if active:
            raise RuntimeError('GPU is occupied; do not interrupt existing work')
        model_identity = core_module('identity')
        manifest = json.loads(args.base_manifest.read_text())
        for name, expected in {**manifest['base_files'], **manifest['tokenizer_files']}.items():
            if digest(args.base/name) != expected:
                raise ValueError('base/tokenizer SHA mismatch')
        if digest(args.adapter/'adapter_model.safetensors') != args.adapter_sha:
            raise ValueError('adapter SHA mismatch')
        inference = core_module('inference')
        tok, model = inference.load_model(str(args.base), str(args.adapter))
        identity = model_identity.verify_loaded(tok, model, base=args.base,
            manifest=args.base_manifest, adapter=args.adapter, adapter_sha=args.adapter_sha)
        generate_real = inference.hf_generator(tok, model, threading.Lock(),
            {'seed': args.seed, 'temperature': .7, 'top_p': .9, 'max_tokens': 256})
        identity.update(model_calls=0, model_loaded=True)
    args.out.mkdir(parents=True, mode=0o700)
    write(args.out/'INPUT_SEAL.json', seal)
    rag = core_module('rag')
    ranker = core_module('retrieval').LocalReranker(args.reranker)
    corpora, results = {}, []
    for case in cases:
        stage = case['stage']
        if stage not in corpora:
            corpora[stage] = rag.StageCorpus(args.script_source,
                [root/'rag'/n for n in ('character_facts.json', 'story_stages.json', 'story_daily_extended.json')],
                stage, reranker=ranker)
        corpus = corpora[stage]
        archived, archived_trace, decisions, active_review = None, None, None, None
        if args.mode == 'replay':
            fixture = fixtures[case['id']]
            if (not isinstance(fixture, dict) or set(fixture) != {'source_support', 'rounds'}
                    or not isinstance(fixture['source_support'], dict)
                    or any(type(value) is not bool for value in fixture['source_support'].values())
                    or not isinstance(fixture['rounds'], list)):
                raise ValueError('replay requires explicit source_support and rounds fixtures')
            decisions = iter(fixture['rounds'])
            key = (case.get('historical_case'), args.seed)
            archived = historical.get(key)
            if case.get('historical_case') and not archived:
                raise ValueError('historical case missing')
            if archived:
                path = args.historical_dir/'traces'/(archived['metadata']['request_id']+'.json')
                archived_trace = json.loads(path.read_text())
                seal[str(path)] = digest(path)
                if archived['request']['messages'][-1]['content'] != case['user']:
                    raise ValueError('replay utterance differs from archived question')
        def generate(messages, *, kind):
            nonlocal active_review
            if generate_real:
                raw = generate_real(messages, kind=kind)
                identity['model_calls'] += 1
                return raw
            if kind == 'plan':
                return (archived_trace['calls'][0]['raw_first'] if archived_trace else
                    json.dumps({'utterance_kind': 'other', 'queries': [], 'candidates': [],
                                'memory_scores': [], 'forget_ids': []}))
            if kind == 'lookup_plan':
                return json.dumps(lookup_fixtures[case['id']], ensure_ascii=False)
            if kind == 'review':
                request = json.loads(messages[-1]['content'])
                if request['mode'] == 'source_support':
                    return json.dumps(fixture['source_support'][request['candidate']['node_id']])
                if active_review is None:
                    active_review = next(decisions)
                if request['mode'] == 'evidence_check':
                    result = active_review['evidence']
                    if result['sufficient']:
                        active_review = None
                elif request['mode'] == 'missing_info':
                    result = active_review['missing']
                    active_review = None
                else:
                    raise ValueError('unknown replay review phase')
                return json.dumps(result, ensure_ascii=False)
            # An old answer is preserved for inspection only. It was generated
            # with a different input. Empty variants mean no role answer exists.
            return archived['final'] if archived else ''
        store = core_module('memory').LifecycleStore(args.out/(case['id']+'.sqlite3'))
        answer, trace = core_module('runtime').Runtime(store, generate,
            search=corpus.search, search_current=corpus.search_current, stage=stage,
            rag_planning=True).reply('fictional-evaluation', 'fixture-event', [
                {'role': 'system', 'content': core_module('prompt').render_system_prompt(
                    (root/'persona/system_prompt.txt').read_text())},
                {'role': 'user', 'content': case['user']}])
        plan = trace['answer_plan']
        required = {r['source_id'] for r in plan['must_respond']}
        expected = set(case['expected_required'])
        checks = {'expected_required_present': expected <= required,
            'no_forbidden_sources': not set(case.get('forbidden_sources', [])) &
                {r['node_id'] for r in trace['retrieved']},
            'unknown_retained': not case.get('expect_unknown') or bool(plan['information_gaps']),
            'bounded_lookup': trace['rag_extra_rounds'] <= 2,
            'no_memory_changes': trace['memory_before'] == trace['memory_after'],
            'valid_arrangement': plan['stop_reason'] not in
                {'invalid_review', 'invalid_lookup_plan', 'planner_no_query'},
            'within_context_budget': len(json.dumps(trace['answer_context'], ensure_ascii=False)) <= 7000}
        record = {'case': case['id'], 'group': case['group'], 'mode': args.mode,
            'reviewer': 'fresh_model' if generate_real else 'scripted_protocol_fixture',
            'second_layer_checks': checks, 'second_layer_pass': all(checks.values()),
            'first_layer_status': 'needs_semantic_and_user_style_review' if generate_real else 'not_run',
            'first_answer': str(answer) if generate_real else None,
            'historical_answer_different_input': str(answer) if archived else None,
            'historical_trace_sha256': digest(path) if archived else None,
            'extra_rounds': trace['rag_extra_rounds'], 'stop_reason': plan['stop_reason'],
            'required_sources': sorted(required)}
        trace['evaluation_provenance'] = {k: record[k] for k in
            ('mode', 'reviewer', 'first_layer_status', 'historical_answer_different_input')}
        write(args.out/(case['id']+'.trace.json'), trace)
        results.append(record)
        print(case['id'], 'protocol_pass=' + str(record['second_layer_pass']), flush=True)
    write(args.out/'INPUT_SEAL.json', seal)
    write(args.out/'identity.json', identity)
    write(args.out/'RESULTS.json', {'mode': args.mode, 'results': results,
        'second_layer_protocol_passes': sum(r['second_layer_pass'] for r in results),
        'cases': len(results), 'fresh_first_layer_cases': len(results) if generate_real else 0,
        'source_hashes': {k: c.source_hashes for k, c in corpora.items()},
        'limitations': [] if generate_real else [
            'Reviewer decisions are fixtures, not model predictions.',
            'Archived first answers used different inputs and are not new-input LoRA results.',
            'Variants test retrieval/program behavior only; fresh model validation is pending.']})


if __name__ == '__main__':
    main()
