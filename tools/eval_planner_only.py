"""Run real, bounded planner/RAG calls on an already loaded base model.

No archived answers, scripted planner decisions, LoRA answer or training are
used. Colab can import this function without downloading/loading the 8B again.
"""
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import threading
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def modules(root):
    """Isolate this frozen source copy from any older notebook imports."""
    source = Path(root) / 'src/atri_core'
    seal = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*.py'))}
    suffix = hashlib.sha256(json.dumps(seal, sort_keys=True).encode()).hexdigest()[:16]
    namespace = '_atri_planner_diagnostic_' + suffix
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(namespace, source / '__init__.py',
                    submodule_search_locations=[str(source)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = package
        spec.loader.exec_module(package)
    return lambda name: importlib.import_module(namespace + '.' + name), seal


def evaluate(tokenizer, model, *, root, script_source, cases_path, out, reranker):
    """Save actual first generations and traces; semantic acceptance is separate."""
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('a new output directory is required')
    if (not getattr(model, 'is_loaded_in_4bit', False) or hasattr(model, 'peft_config')
            or getattr(model.config, 'model_type', None) != 'qwen3'):
        raise ValueError('this diagnostic requires the original NF4 Qwen3 base without a LoRA')
    quant = model.config.quantization_config
    quant = quant.to_dict() if hasattr(quant, 'to_dict') else quant
    if any(quant.get(key) != value for key, value in {
            'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_use_double_quant': False}.items()):
        raise ValueError('existing model quantization differs from the original settings')
    core, source_seal = modules(root)
    cases = [json.loads(line) for line in Path(cases_path).read_text().splitlines() if line.strip()]
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('cases require distinct IDs')
    out.mkdir(parents=True, mode=0o700)
    card_files = [root / 'rag' / name for name in
                  ('character_facts.json', 'story_stages.json', 'story_daily_extended.json')]
    write(out / 'INPUT_SEAL.json', {'source': source_seal, 'cases': digest(cases_path),
        'cards': {p.name: digest(p) for p in card_files},
        'evaluator': digest(Path(__file__))})
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'sentence-transformers'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import torch
    write(out / 'ENVIRONMENT.json', {'versions': versions, 'python': sys.version,
        'os': platform.platform(), 'gpu': torch.cuda.get_device_name(0),
        'quantization': quant, 'generation_config': model.generation_config.to_dict(),
        'seed': 17, 'enable_thinking': False, 'adapter_loaded': False})
    model.eval()
    real_generate = core('inference').hf_generator(tokenizer, model, threading.Lock(),
        {'seed': 17, 'temperature': .7, 'top_p': .9, 'max_tokens': 256}, base_only=True)
    active, call_count = None, 0

    def generate(messages, *, kind):
        nonlocal call_count
        if kind == 'answer':
            raise AssertionError('planner diagnostic must never request a role answer')
        append(out / 'requests.jsonl', {'request_id': call_count, 'case': active,
                                      'kind': kind, 'messages': messages})
        raw = real_generate(messages, kind=kind)
        record = {'request_id': call_count, 'case': active, 'kind': kind, 'messages': messages,
                  'raw_first': str(raw), 'generation': raw.metadata}
        append(out / 'raw-model.jsonl', record)
        call_count += 1
        return raw

    results, corpora = [], {}
    try:
        for case in cases:
            active, stage = case['id'], case['stage']
            if stage not in corpora:
                corpora[stage] = core('rag').StageCorpus(script_source, card_files, stage,
                                                       reranker=reranker, threshold=.5, top_k=3)
            corpus = corpora[stage]
            store = core('memory').LifecycleStore(out / (active + '.sqlite3'))
            answer, trace = core('runtime').Runtime(store, generate, search=corpus.search,
                search_current=corpus.search_current, stage=stage, rag_planning=True).reply(
                    'fictional-planner-evaluation', 'single-event',
                    [{'role': 'user', 'content': case['user']}], generate_answer=False)
            assert answer is None
            plan = trace['answer_plan']
            required = {fact['source_id'] for fact in plan['must_respond']}
            checks = {
                'needs_declared': bool(plan['information_needs']),
                'expected_required_present': set(case['expected_required']) <= required,
                'no_forbidden_sources': not set(case.get('forbidden_sources', [])) &
                    {hit['node_id'] for hit in trace['retrieved']},
                'unknown_retained': not case.get('expect_unknown') or bool(plan['information_gaps']),
                'bounded_lookup': trace['rag_extra_rounds'] <= 2,
                'no_memory_changes': trace['memory_before'] == trace['memory_after'],
                'valid_arrangement': plan['stop_reason'] not in
                    {'invalid_review', 'invalid_lookup_plan', 'planner_no_query'},
                'no_retrieval_failure': not any(error['phase'] == 'rag_lookup' for error in trace['errors']),
                'within_context_budget': len(json.dumps(trace['answer_context'], ensure_ascii=False)) <= 7000,
            }
            row = {'case': active, 'group': case['group'], 'question': case['user'],
                'stage': stage, 'program_checks': checks, 'program_checks_pass': all(checks.values()),
                'needs': plan['information_needs'], 'queries': trace['retrieval_queries'],
                'required_sources': sorted(required), 'information_gaps': plan['information_gaps'],
                'extra_rounds': trace['rag_extra_rounds'], 'stop_reason': plan['stop_reason'],
                'model_calls': len(trace['calls']), 'planner_semantic_review': 'pending',
                'role_answer': 'not_run', 'role_tone': 'not_run'}
            trace['evaluation_provenance'] = {'mode': 'real_planner_only',
                'fixtures_used': False, 'historical_answers_used': False, 'role_generation': 'not_run'}
            write(out / (active + '.trace.json'), trace)
            results.append(row)
            write(out / 'RESULTS.json', {'state': 'running', 'results': results,
                'completed_cases': len(results), 'model_calls': call_count, 'role_answers': 0})
            print(active, '程序检查=' + str(row['program_checks_pass']),
                  '停止原因=' + plan['stop_reason'], '查询=' + str(row['queries']), flush=True)
    except Exception:
        write(out / 'FAILURE.json', {'case': active, 'traceback': traceback.format_exc(),
            'completed_cases': len(results), 'model_calls': call_count})
        raise
    report = {'state': 'complete', 'results': results, 'completed_cases': len(results),
        'program_checks_passes': sum(row['program_checks_pass'] for row in results),
        'model_calls': call_count, 'role_answers': 0, 'training_steps': 0,
        'source_hashes': {stage: corpus.source_hashes for stage, corpus in corpora.items()},
        'limitation': 'Program checks do not prove query semantics or factual completeness. Role/LoRA generation was not run.'}
    write(out / 'RESULTS.json', report)
    return report
