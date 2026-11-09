"""Isolated OpenAI-compatible transport for the shared V1.2 development runtime.

Explicit configuration only. This does not replace the existing service/QQ bot.
Run with --help for required paths. Model loading remains in the existing loader.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .auth import MemoryService
from .memory import LifecycleStore, Policy
from .runtime import PROFILE, Runtime


from .files import file_sha


def build_app(store, users, generator_factory, *, system, identity, search=None, search_current=None,
              stage='daily', trace_dir=None, principal_resolver=None, rag_planning=False):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from .prompt import render_system_prompt

    system = render_system_prompt(system)
    identity = {**identity, 'system_prompt_sha256': hashlib.sha256(system.encode()).hexdigest(),
                'rag_planning': rag_planning}
    app = FastAPI()
    auth = MemoryService(None, users) if principal_resolver is None else None
    # The model and request pipeline are serialized together, including planning,
    # writes, and generation. A later request cannot observe half a conversation.
    pipeline_lock = threading.Lock()
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True, mode=0o700)

    @app.get('/health')
    def health():
        return {'ok': True, 'profile': PROFILE, 'identity': identity, 'stage': stage}

    @app.get('/v1/models')
    def models():
        return {'object': 'list', 'data': [{'id': identity['model'], 'object': 'model'}]}

    @app.get('/v1/memory')
    def memory(request: Request):
        if auth is None:
            raise HTTPException(403, 'use the authenticated platform memory entry')
        owner = auth.owner(request.headers.get('authorization'))
        return {'items': store.records(owner)}

    @app.delete('/v1/memory')
    def forget(request: Request, value: str | None = None):
        if auth is None:
            raise HTTPException(403, 'use the authenticated platform memory entry')
        owner = auth.owner(request.headers.get('authorization')) if auth else None
        with pipeline_lock:
            return store.forget(owner, value=value)

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        owner = auth.owner(request.headers.get('authorization')) if auth else None
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, 'object required')
        incoming = body.get('messages')
        event_id = request.headers.get('idempotency-key')
        trusted_notice = ''
        if principal_resolver is not None:
            owner, event_id, incoming, trusted_notice = principal_resolver(body)
        if not isinstance(incoming, list) or not incoming or len(incoming) > 64:
            raise HTTPException(400, '1–64 messages required')
        if any(not isinstance(m, dict) or m.get('role') not in {'user', 'assistant'}
               or not isinstance(m.get('content'), str) or len(m['content']) > 12000 for m in incoming):
            raise HTTPException(400, 'text user/assistant messages required; persona is server configured')
        if incoming[-1]['role'] != 'user':
            raise HTTPException(400, 'current user turn required')
        if not event_id or len(event_id) > 220:
            raise HTTPException(400, 'Idempotency-Key with a stable user event ID required')
        # Request user, owner, stage, system, and model fields cannot select a
        # different principal, persona, stage, or loaded adapter.
        messages = [{'role': 'system', 'content': system + trusted_notice}, *[
            {'role': m['role'], 'content': m['content']} for m in incoming]]
        rid = 'chatcmpl-' + uuid.uuid4().hex
        def execute():
            with pipeline_lock:
                runtime = Runtime(store, generator_factory(body), search=search,
                                  search_current=search_current, stage=stage, rag_planning=rag_planning)
                answer, trace = runtime.reply(owner, event_id, messages)
                trace.update({'request_id': rid, 'identity': identity, 'event_id': event_id})
                if trace_dir:
                    target = Path(trace_dir) / (rid + '.json')
                    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, 'w') as dest:
                        json.dump(trace, dest, ensure_ascii=False, indent=2)
                return answer, trace
        try:
            answer, trace = await asyncio.to_thread(execute)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        common = {'id': rid, 'model': identity['model'], 'created': int(time.time())}
        details = {'profile': PROFILE, 'request_id': rid, 'calls': len(trace['calls']),
                   'operations': trace['operations'], 'errors': trace['errors']}
        finish = trace['calls'][-1].get('generation', {}).get('finish_reason', 'stop')
        if not body.get('stream'):
            return {**common, 'object': 'chat.completion', 'choices': [{'index': 0,
                'message': {'role': 'assistant', 'content': answer}, 'finish_reason': finish}], 'atri': details}
        async def stream():
            # Buffered transport: exactly the same first answer as JSON. No
            # repair or fallback can silently change the streamed contents.
            for offset in range(0, len(answer), 32):
                delta = {'content': answer[offset:offset+32]}
                if offset == 0:
                    delta['role'] = 'assistant'
                chunk = {**common, 'object': 'chat.completion.chunk', 'choices': [
                    {'index': 0, 'delta': delta, 'finish_reason': None}]}
                yield 'data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n'
            yield 'data: ' + json.dumps({**common, 'object': 'chat.completion.chunk', 'choices': [
                {'index': 0, 'delta': {}, 'finish_reason': finish}], 'atri': details}, ensure_ascii=False) + '\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(stream(), media_type='text/event-stream')
    return app


def main():
    from .config import load_runtime_config
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config', type=Path)
    selected, _ = preliminary.parse_known_args()
    defaults = load_runtime_config(selected.config) if selected.config else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='private JSON config; command-line values override it')
    for name in ('base', 'adapter', 'system', 'database', 'traces'):
        parser.add_argument('--' + name, type=Path, required=name not in defaults)
    principal = parser.add_mutually_exclusive_group(required=not (defaults.get('users') or defaults.get('qq_secret')))
    principal.add_argument('--users',type=Path,help='isolated owner-to-Bearer-token JSON')
    principal.add_argument('--qq-secret',type=Path,help='signed private QQ entrance; no Bearer map')
    parser.add_argument('--qq-context-store',type=Path,help='optional PR #17 ContextWindow DB; requires atri_context module')
    parser.add_argument('--adapter-sha', required='adapter_sha' not in defaults)
    parser.add_argument('--base-manifest', type=Path, required='base_manifest' not in defaults)
    parser.add_argument('--memory-policy', type=Path)
    parser.add_argument('--planner-adapter', action='store_true', help='also use the role LoRA for internal planning')
    parser.add_argument('--planner-schema', action='store_true', help='experimental token-level JSON planning constraint')
    parser.add_argument('--rag-planning', action='store_true', help='experimental source-bound answer plan and at most two extra RAG rounds')
    parser.add_argument('--stage', choices=['daily', 'school_later', 'eden_partial', 'past_recovered'], default='daily')
    parser.add_argument('--script-source', type=Path)
    parser.add_argument('--story-cards', type=Path, nargs='+')
    parser.add_argument('--story-dense-model', help='optional local BGE Chinese model for supplementary retrieval')
    parser.add_argument('--story-dense-threshold', type=float, help='explicit corpus-calibrated cosine threshold')
    parser.add_argument('--port', type=int, default=8022)
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    if bool(args.users) == bool(args.qq_secret):
        parser.error('choose exactly one authentication mode in the config or command line')
    from .rag import validate_dense_settings
    validate_dense_settings(args.story_dense_model, args.story_dense_threshold)
    if bool(args.story_cards) != bool(args.script_source):
        raise ValueError('--story-cards and --script-source are required together')
    if args.story_dense_model is not None and not args.story_cards:
        raise ValueError('dense story retrieval requires --story-cards and --script-source')
    adapter_sha = file_sha(args.adapter / 'adapter_model.safetensors')
    if adapter_sha != args.adapter_sha:
        raise ValueError('adapter SHA mismatch')
    manifest = json.loads(args.base_manifest.read_text())
    for name, expected in {**manifest['base_files'], **manifest['tokenizer_files']}.items():
        if file_sha(args.base/name) != expected:
            raise ValueError('base/tokenizer SHA mismatch: ' + name)
    from .inference import load_model
    from .identity import verify_loaded
    tok, model = load_model(str(args.base), str(args.adapter))
    model_lock = threading.Lock()
    verified = verify_loaded(tok, model, base=args.base, manifest=args.base_manifest,
                             adapter=args.adapter, adapter_sha=args.adapter_sha)
    from importlib.metadata import version
    identity = {'model': 'Qwen3-8B+adapter-' + adapter_sha[:12], 'base_files': {
        p.name: file_sha(p) for p in sorted(args.base.glob('*'))
        if p.suffix in {'.json', '.safetensors'} and p.is_file()},
        'adapter_sha256': adapter_sha, 'adapter_config_sha256': file_sha(args.adapter/'adapter_config.json'),
        'loaded_adapter_tensors_exact': verified['loaded_tensors_exact'],
        'planner_adapter_enabled': args.planner_adapter, 'answer_adapter_enabled': True,
        'planner_schema_enabled': args.planner_schema,
        'enable_thinking': False, 'quantization': model.config.quantization_config.to_dict(),
        'versions': {p: version(p) for p in ['torch', 'transformers', 'peft', 'bitsandbytes']},
        'chat_template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest()}
    # Exact loaded tensor equality is recorded; it is not semantic acceptance.
    def factory(body):
        from .inference import hf_generator
        return hf_generator(tok, model, model_lock, body, planner_adapter=args.planner_adapter,
                            planner_schema=args.planner_schema)
    search = search_current = None
    if args.story_cards:
        from .retrieval import Config, LocalReranker
        from .rag import StageCorpus, LocalDenseScorer
        corpus = StageCorpus(args.script_source, args.story_cards, args.stage,
            reranker=LocalReranker(Config().rerank_model),
            dense_scorer=LocalDenseScorer(args.story_dense_model) if args.story_dense_model else None,
            dense_threshold=args.story_dense_threshold)
        identity['story_source_hashes'] = corpus.source_hashes
        identity['story_dense'] = {'model': args.story_dense_model, 'threshold': args.story_dense_threshold}
        search = corpus.search
        search_current = corpus.search_current
    store = LifecycleStore(args.database, Policy(**json.loads(args.memory_policy.read_text())) if args.memory_policy else Policy())
    options=dict(system=args.system.read_text(),identity=identity,search=search,search_current=search_current,
                 stage=args.stage,trace_dir=args.traces,rag_planning=args.rag_planning)
    if args.qq_secret:
        from .bridge import read_secret
        from .qq import build_qq_app
        context=None
        if args.qq_context_store:
            from .context import ContextWindow
            context=ContextWindow(args.qq_context_store)
        app=build_qq_app(store,read_secret(args.qq_secret),factory,context_window=context,**options)
    else:
        if args.qq_context_store:
            raise ValueError('--qq-context-store requires --qq-secret')
        app=build_app(store,json.loads(args.users.read_text()),factory,**options)
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
