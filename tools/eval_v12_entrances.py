"""Real-model entrance checks after development review, using fictional state.

Runs native ASGI handlers, signed QQ envelopes, Colab mounting and native web
JWT/SSE. This is not live QQ delivery, Google Colab hosting or PostgreSQL testing.
No automatic semantic pass; every first output remains available for review.
"""
import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from atri_memory_lifecycle import Candidate, LifecycleStore
from atri_model_identity import verify_loaded
from serve_memory_rag import build_app, file_sha

CASES = ('dev.d.important','dev.d.correct','dev.d.forget','dev.c.walk',
         'dev.c.taste','dev.d.failure','dev.d.isolation','dev.b.daily')


def fixture(store, owner, records):
    for i,row in enumerate(records):
        store.observe(owner,'fixture:'+str(i),row['statement'],[Candidate(
            row['subject'],row['predicate'],row['value'],row['statement'],
            polarity=row['polarity'],degree=row['degree'])])


def openai_result(response, stream):
    if response.status_code != 200:
        raise ValueError(f'HTTP {response.status_code}: {response.text[:300]}')
    if not stream:
        result=response.json()
        return result['choices'][0]['message']['content'],result.get('atri')
    if not response.text.rstrip().endswith('data: [DONE]'):
        raise ValueError('incomplete SSE')
    chunks=[json.loads(s[6:]) for s in response.text.splitlines() if s.startswith('data: {')]
    return ''.join(c['choices'][0]['delta'].get('content','') for c in chunks),chunks[-1].get('atri')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','adapter','base-manifest','system','seal','cases','script-source','out','development-review'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--adapter-sha',required=True)
    args=parser.parse_args()
    seal=json.loads(args.seal.read_text())
    if args.cases.name != 'development.jsonl' or file_sha(args.cases)!=seal['files'][args.cases.name]['sha256']:
        raise ValueError('sealed development file required')
    review=json.loads(args.development_review.read_text())
    if (review.get('ready_for_entrance_validation') is not True
            or review.get('adapter_sha256')!=args.adapter_sha
            or review.get('dataset_sha256')!=file_sha(args.cases)):
        raise ValueError('matching completed development review required before this batch')
    rows={r['id']:r for r in map(json.loads,args.cases.read_text().splitlines())}
    cases=[rows[k] for k in CASES]
    if any(r['stage']!='daily' for r in cases):
        raise ValueError('this entrance batch uses daily-stage cases')
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import serve_v1_1_8b as loader
    loader.BASE,loader.ADAPTER,loader.SYSTEM_FILE=str(args.base),str(args.adapter),str(args.system)
    loader.load()
    tok,model=loader.state['tok'],loader.state['model']
    identity=verify_loaded(tok,model,base=args.base,manifest=args.base_manifest,
                           adapter=args.adapter,adapter_sha=args.adapter_sha)
    identity.update(dataset_sha256=file_sha(args.cases),development_review_sha256=file_sha(args.development_review),
                    entrance_script_sha256=file_sha(__file__),hosting_scope=__doc__,
                    chat_template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                    versions={p:version(p) for p in ('torch','transformers','peft','bitsandbytes',
                        'fastapi','sentence-transformers','llama-index-core')})
    from atri_generation import hf_generator
    from atri_story import StageCorpus
    from atri_rag import Config,LocalReranker
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from atri_bridge import attach_context
    from atri_qq_v12 import attach_event,build_qq_app
    root=Path(__file__).resolve().parents[1]
    identity['source_sha256']={str(p.relative_to(root)):file_sha(p)
        for folder in ('tools','backend/app','colab') for p in (root/folder).rglob('*.py')}
    corpus=StageCorpus(args.script_source,[root/'rag'/n for n in
        ('character_facts.json','story_stages.json','story_daily_extended.json')],
        'daily',reranker=LocalReranker(Config().rerank_model))
    identity['story_source_hashes']=corpus.source_hashes
    (args.out/'identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2))
    factory=lambda body:hf_generator(tok,model,loader.lock,body)
    system=loader.default_system()
    secret=b'fictional-entrance-validation-only-bridge-secret'
    total=0
    for entrance in ('http-json','http-sse','qq-json','qq-sse','colab-json','colab-sse','web-sse'):
        folder=args.out/entrance;folder.mkdir()
        store=LifecycleStore(folder/'memory.sqlite3')
        owners={r['id']:('web:'+str(i) if entrance=='web-sse' else
               'qq:isolated:'+str(i) if entrance.startswith('qq') else 'fictional:'+r['id'])
                for i,r in enumerate(cases,1)}
        users={owner:hashlib.sha256(('fictional-entrance:'+owner).encode()).hexdigest() for owner in owners.values()}
        common=dict(system=system,identity=identity,search=corpus.search,trace_dir=folder/'traces')
        web=None
        if entrance.startswith('qq'):
            app=build_qq_app(store,secret,factory,**common)
        elif entrance.startswith('colab'):
            spec=importlib.util.spec_from_file_location('colab_v12_runtime',root/'colab/v12_runtime.py')
            colab=importlib.util.module_from_spec(spec);spec.loader.exec_module(colab)
            app=FastAPI()
            colab.mount_runtime(app,tokenizer=tok,model=model,base_path=args.base,base_manifest=args.base_manifest,
                adapter_path=args.adapter,expected_adapter_sha=args.adapter_sha,users=users,database=store.path,
                system=system,search=corpus.search,trace_dir=folder/'traces',model_lock=loader.lock)
        elif entrance=='web-sse':
            web=make_web(root,folder,store,factory,identity,corpus.search,args.system,len(cases))
            app=web['app']
        else:
            app=build_app(store,users,factory,**common)
        with TestClient(app) as client:
            for number,row in enumerate(cases,1):
                owner=owners[row['id']]
                fixture(store,owner,row['setup'])
                fixture(store,owner+':other',row.get('other_user_setup',[]))
                if row.get('fail_save'):
                    with store.connect() as db:
                        db.execute("CREATE TRIGGER fail_eval BEFORE INSERT ON facts BEGIN SELECT RAISE(ABORT,'injected failure'); END")
                history=[];conversation=None
                for turn_index,turn in enumerate(row['turns']):
                    if turn_index in row.get('reset_before_turns',[]):
                        history=[];conversation=None
                    history.append({'role':'user','content':turn['user']})
                    event=f'{row["id"]}:{turn_index}'
                    body={'messages':list(history),'seed':17,'max_tokens':256,'temperature':.7,'top_p':.9,
                          'stream':entrance.endswith('sse'),'owner':owner+':other','user':owner+':other'}
                    if entrance.startswith('qq'):
                        qq=SimpleNamespace(message_obj=SimpleNamespace(message_id=event),get_platform_id=lambda:'isolated',
                            get_sender_id=lambda:str(number),is_private_chat=lambda:True)
                        signed=attach_event(attach_context('',owner,[],True,secret),qq,turn['user'],secret)
                        body['messages']=[{'role':'system','content':signed},*history]
                        response=client.post('/v1/chat/completions',json=body)
                    elif web:
                        body={'conversation_id':conversation,'content':turn['user'],'enable_thinking':False,'user_id':99999}
                        response=client.post('/api/chat/send',headers=web['headers'](number),json=body)
                    else:
                        prefix='/v12' if entrance.startswith('colab') else ''
                        response=client.post(prefix+'/v1/chat/completions',
                            headers={'Authorization':'Bearer '+users[owner],'Idempotency-Key':event},json=body)
                    if web:
                        if response.status_code!=200:raise ValueError(response.text[:300])
                        chunks=[json.loads(s[6:]) for s in response.text.splitlines() if s.startswith('data: ')]
                        if not chunks[-1].get('done'):raise ValueError('incomplete native web SSE')
                        answer=''.join(c['content'] for c in chunks)
                        conversation=chunks[-1]['conversation_id'];metadata=chunks[-1].get('atri')
                    else:
                        answer,metadata=openai_result(response,body['stream'])
                    history.append({'role':'assistant','content':answer})
                    result={'entrance':entrance,'case':row['id'],'turn':turn_index,'request':body,'final':answer,
                            'program_state':store.records(owner,active_only=False),'metadata':metadata,
                            'expected_semantics':turn['expect'],'semantic_review':'pending'}
                    with (folder/'results.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                    total+=1
                    print(entrance,row['id'],turn_index,flush=True)
                if row.get('fail_save'):
                    with store.connect() as db:db.execute('DROP TRIGGER fail_eval')
        if web:web['engine'].dispose()
    (args.out/'completion.json').write_text(json.dumps({'http_turns':total,'scenarios_per_transport':len(cases),
        'semantic_review':'pending','v1_2_accepted':False},indent=2))


def make_web(root,folder,store,factory,identity,search,system,user_count):
    """Use real JWT dependency and native routes with isolated SQLite schema."""
    sys.path.insert(0,str(root/'backend'))
    os.environ.update(DATABASE_URL='sqlite:///'+str(folder/'web.sqlite3'),
                      SECRET_KEY='fictional-entrance-evaluation-only-key',SYSTEM_PROMPT_PATH=str(system))
    from fastapi import FastAPI
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    @compiles(JSONB,'sqlite')
    def sqlite_json(type_,compiler,**kw):return 'JSON'
    from app.db import Base,get_db
    from app.models import User
    from app.routers.chat import router
    from app.security import create_access_token
    from app.v12_runtime import install
    engine=create_engine(os.environ['DATABASE_URL'],connect_args={'check_same_thread':False})
    Base.metadata.create_all(engine);sessions=sessionmaker(bind=engine)
    with sessions() as db:
        db.add_all(User(id=i,email=f'fictional{i}@example.test',password_hash='unused',display_name=f'用户{i}')
                   for i in range(1,user_count+1));db.commit()
    def session():
        with sessions() as db:yield db
    app=FastAPI();app.include_router(router);app.dependency_overrides[get_db]=session
    install(app,store,factory,identity=identity,search=search,trace_dir=folder/'traces')
    return {'app':app,'engine':engine,'headers':lambda i:{'Authorization':'Bearer '+create_access_token(i)}}


if __name__=='__main__':main()
