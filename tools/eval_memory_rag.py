"""Real 8B, first-output comparison through original and new HTTP handlers.

Requires a sealed scenario file and an independently recorded base manifest.
No training, no production stores, no external generation, no automatic grading.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['base', 'adapter', 'base-manifest', 'system', 'seal', 'cases', 'baseline-index', 'script-source', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--adapter-sha', required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17,43,101])
    parser.add_argument('--allow-final', action='store_true', help='only after development gate; consumes the holdout')
    parser.add_argument('--sse', action='store_true')
    parser.add_argument('--planner-adapter', action='store_true')
    parser.add_argument('--planner-schema', action='store_true')
    parser.add_argument('--story-dense-model')
    parser.add_argument('--story-dense-threshold', type=float)
    args = parser.parse_args()
    from atri_story import validate_dense_settings
    validate_dense_settings(args.story_dense_model, args.story_dense_threshold)
    seal = json.loads(args.seal.read_text())
    assert args.cases.name in seal['files']
    assert digest(args.cases) == seal['files'][args.cases.name]['sha256']
    assert set(args.seeds) <= set(seal['seeds'])
    if 'final' in args.cases.name and not args.allow_final:
        raise ValueError('holdout is sealed; --allow-final explicitly consumes it')
    rows = [json.loads(line) for line in args.cases.read_text().splitlines()]
    if args.ids:
        assert set(args.ids) <= {r['id'] for r in rows}
        rows = [r for r in rows if r['id'] in args.ids]
    assert rows and digest(args.adapter/'adapter_model.safetensors') == args.adapter_sha
    base_manifest = json.loads(args.base_manifest.read_text())
    verified = {}
    for filename, expected in {**base_manifest['base_files'], **base_manifest['tokenizer_files']}.items():
        actual = digest(args.base/filename)
        assert actual == expected, (filename, 'base/tokenizer identity mismatch')
        verified[filename] = actual
    args.out.mkdir(parents=True, exist_ok=False)
    users = {}
    for row in rows:
        for seed in args.seeds:
            for suffix in ['', ':other']:
                owner = f"fictional:{row['id']}:{seed}{suffix}"
                users[owner] = hashlib.sha256(('isolated-eval-only:'+owner).encode()).hexdigest()
    users_file = args.out/'fictional-users.json'
    users_file.write_text(json.dumps(users)); users_file.chmod(0o600)
    os.environ.update({'ATRI_BASE':str(args.base), 'ATRI_ADAPTER':str(args.adapter),
        'ATRI_SYSTEM':str(args.system), 'ATRI_MEMORY_ENABLED':'1',
        'ATRI_MEMORY_DB':str(args.out/'original-memory.sqlite3'), 'ATRI_MEMORY_USERS_FILE':str(users_file),
        'ATRI_RAG_ENABLED':'1', 'ATRI_RAG_STORAGE':str(args.baseline_index), 'ATRI_GROUNDING_ENABLED':'1',
        'HF_HUB_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false'})
    import serve_v1_1_8b as original
    original.load()
    import torch
    from safetensors.torch import load_file
    from fastapi.testclient import TestClient
    from importlib.metadata import version
    from atri_memory_lifecycle import Candidate, LifecycleStore
    from atri_story import StageCorpus, LocalDenseScorer
    from atri_rag import Config, LocalReranker
    from serve_memory_rag import build_app
    tok, model = original.state['tok'], original.state['model']
    source = load_file(args.adapter/'adapter_model.safetensors')
    observed = {}
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            key = name.replace('.default.', '.')
            assert key in source and torch.equal(parameter.detach().cpu(), source[key]), (name,'loaded tensor differs')
            observed[key] = str(parameter.dtype)
    assert set(source) == set(observed)
    del source
    identity = {'model':'Qwen3-8B+adapter-'+args.adapter_sha[:12], 'adapter_sha256':args.adapter_sha,
        'loaded_tensors_exact':len(observed), 'adapter_dtypes':sorted(set(observed.values())),
        'planner_adapter_enabled':args.planner_adapter, 'answer_adapter_enabled':True,
        'planner_schema_enabled':args.planner_schema,
        'story_dense': {'model': args.story_dense_model, 'threshold': args.story_dense_threshold},
        'base_and_tokenizer_sha256':verified, 'enable_thinking':False,
        'chat_template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        'quantization':model.config.quantization_config.to_dict(), 'default_generation':model.generation_config.to_dict(),
        'versions':{p:version(p) for p in ['torch','transformers','peft','bitsandbytes','llama-index-core','sentence-transformers']},
        'source_hashes':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        'dataset_sha256':digest(args.cases), 'system_sha256':digest(args.system),
        'baseline_rag_manifest_sha256':digest(args.baseline_index/'atri_manifest.json'),
        'comparison_differences':['new automatic extraction and lifecycle','new model planning call',
          'new stage-filtered source summaries and relevance selection','original repair/fallback versus new raw answer'],
        'not_automatically_accepted':True, 'started_unix':time.time()}
    (args.out/'identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2))
    calls, active, pending = [], {}, [0]
    condition = threading.Condition()
    original_generate = model.generate
    def captured_generate(*positional, **kwargs):
        with condition:
            pending[0] += 1
        ids = kwargs.get('input_ids', positional[0] if positional else None)
        prompt_ids = ids[0].detach().cpu().tolist()
        context = dict(active)
        started = time.time()
        try:
            output = original_generate(*positional, **kwargs)
            output_ids = output[0,len(prompt_ids):].detach().cpu().tolist()
            record = {**context, 'prompt_token_ids':prompt_ids, 'output_token_ids':output_ids,
                'model_prompt':tok.decode(prompt_ids), 'raw_first':tok.decode(output_ids,skip_special_tokens=True),
                'seconds':time.time()-started, 'generation':{k:v for k,v in kwargs.items()
                    if k in {'max_new_tokens','do_sample','temperature','top_p','top_k','repetition_penalty','pad_token_id'}},
                'hit_token_limit':len(output_ids)>=kwargs.get('max_new_tokens',10**9)}
            with condition:
                record['call_index']=len(calls)
                calls.append(record)
                with (args.out/'raw-model.jsonl').open('a') as f:
                    f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush()
            return output
        finally:
            with condition:
                pending[0] -= 1
                condition.notify_all()
    model.generate = captured_generate
    def factory(body):
        from atri_generation import hf_generator
        implementation = hf_generator(tok, model, original.lock, body, planner_adapter=args.planner_adapter,
                                      planner_schema=args.planner_schema)
        def generate(messages, *, kind):
            active['kind']=kind
            return implementation(messages,kind=kind)
        return generate
    store=LifecycleStore(args.out/'improved-memory.sqlite3')
    reranker=LocalReranker(Config().rerank_model)
    dense=LocalDenseScorer(args.story_dense_model) if args.story_dense_model else None
    card_files=[Path(__file__).resolve().parents[1]/'rag'/name for name in
                ['character_facts.json','story_stages.json','story_daily_extended.json']]
    clients={}
    for stage in sorted({r['stage'] for r in rows}):
        corpus=StageCorpus(args.script_source,card_files,stage,reranker=reranker,
                          dense_scorer=dense,dense_threshold=args.story_dense_threshold)
        clients[stage]=TestClient(build_app(store,users,factory,system=original.default_system(),
            identity=identity,search=corpus.search,search_current=corpus.search_current,
            stage=stage,trace_dir=args.out/'traces'))
    legacy=TestClient(original.build_app())
    all_http=[]
    for row in rows:
        for seed in args.seeds:
            owner=f"fictional:{row['id']}:{seed}"
            for target, fixtures in [(owner,row['setup']),(owner+':other',row.get('other_user_setup',[]))]:
                for i,item in enumerate(fixtures):
                    response=legacy.put('/v1/memory',headers={'Authorization':'Bearer '+users[target]},
                                        json={'key':item['key'],'value':item['value']})
                    assert response.status_code==200
                    candidate=Candidate(item['subject'],item['predicate'],item['value'],item['statement'],
                                        polarity=item['polarity'],degree=item['degree'])
                    store.observe(target,'fixture:'+str(i),item['statement'],[candidate])
            if row.get('fail_save'):
                with store.connect() as db:
                    # Owner is a generated fixture identifier, not model/user SQL.
                    db.execute("CREATE TRIGGER fail_eval BEFORE INSERT ON facts WHEN NEW.owner="+
                        "'"+owner.replace("'","''")+"' BEGIN SELECT RAISE(ABORT,'injected persistence failure'); END")
            for arm,client in [('original',legacy),('improved',clients[row['stage']])]:
                history=[]
                for turn_index, turn in enumerate(row['turns']):
                    if turn_index in row.get('reset_before_turns',[]):
                        history=[]
                        # Each store method opens a fresh SQLite connection;
                        # also reconstruct the store to exercise persistence.
                        if arm=='improved':
                            assert LifecycleStore(store.path).records(owner)==store.records(owner)
                    history.append({'role':'user','content':turn['user']})
                    body={'messages':list(history),'seed':seed,'temperature':.7,'top_p':.9,
                          'max_tokens':256,'stream':args.sse}
                    if row.get('spoof_body_owner'):
                        body.update(owner=owner+':other',user=owner+':other')
                    active.clear();active.update(case=row['id'],arm=arm,seed=seed,turn=turn_index,kind='original')
                    start=len(calls)
                    response=client.post('/v1/chat/completions',headers={'Authorization':'Bearer '+users[owner],
                        'Idempotency-Key':f'{row["id"]}:{seed}:{turn_index}'},json=body)
                    with condition:
                        assert condition.wait_for(lambda: pending[0]==0,timeout=300), 'generation capture still pending'
                    assert response.status_code==200, (arm,row['id'],response.status_code,response.text[:500])
                    if args.sse:
                        chunks=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
                        assert response.text.rstrip().endswith('data: [DONE]')
                        answer=''.join(c['choices'][0]['delta'].get('content','') for c in chunks)
                        metadata=chunks[-1].get('atri')
                    else:
                        payload=response.json();answer=payload['choices'][0]['message']['content'];metadata=payload.get('atri')
                    history.append({'role':'assistant','content':answer})
                    state=(legacy.get('/v1/memory',headers={'Authorization':'Bearer '+users[owner]}).json()
                           if arm=='original' else {'items':store.records(owner,active_only=False)})
                    result={**dict(active),'category':row['category'],'stage':row['stage'],'request':body,
                        'final':answer,'raw_call_indices':list(range(start,len(calls))),'program_state':state,
                        'metadata':metadata,'expected_semantics':turn['expect'],'semantic_review':'pending'}
                    with (args.out/'http-results.jsonl').open('a') as f:
                        f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                    all_http.append(result)
                    print(row['id'],seed,arm,turn_index,'calls',len(calls)-start,flush=True)
            if row.get('fail_save'):
                with store.connect() as db:db.execute('DROP TRIGGER fail_eval')
    assert digest(args.adapter/'adapter_model.safetensors')==args.adapter_sha
    (args.out/'completion.json').write_text(json.dumps({'finished_unix':time.time(),
        'http_turns':len(all_http),'model_calls':len(calls),'adapter_unchanged':True,
        'semantic_review':'pending','v1_2_accepted':False},indent=2))


if __name__=='__main__':
    main()
