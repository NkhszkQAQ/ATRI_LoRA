"""Native backend handler tests using fictional SQLite state and real JWT auth.

SQLite compilation is test-only; these tests do not claim PostgreSQL validation.
"""
import json
import os
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'tools')]
os.environ['DATABASE_URL']='sqlite:///:memory:'
os.environ['SECRET_KEY']='fictional-v12-web-test-key-not-for-deployment'
os.environ['SYSTEM_PROMPT_PATH']=str(ROOT/'persona/system_prompt.txt')

@compiles(JSONB,'sqlite')
def jsonb_sqlite(type_,compiler,**kw):
    return 'JSON'

from app.db import Base,get_db
from app.models import User,Message
from app.routers.chat import router
from app.security import create_access_token
from app.v12_runtime import install
from atri_memory_lifecycle import LifecycleStore


@pytest.fixture
def web(tmp_path):
    engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions=sessionmaker(bind=engine)
    with sessions() as db:
        db.add_all([User(id=i,email=f'fictional{i}@example.test',password_hash='unused',
            display_name=f'用户{i}',memory_note='我喜欢旧库秘密饮品') for i in [1,2]])
        db.commit()
    app=FastAPI();app.include_router(router)
    def session():
        with sessions() as db:yield db
    app.dependency_overrides[get_db]=session
    calls=[]
    def factory(options):
        def generate(messages,*,kind):
            calls.append(messages)
            return '{"utterance_kind":"self_report"}' if kind=='plan' else '这是首次回答。'
        return generate
    store=LifecycleStore(tmp_path/'v12.db')
    runtime=install(app,store,factory,identity={'model':'test-double'},trace_dir=tmp_path/'traces')
    with TestClient(app) as client:
        yield client,store,sessions,calls,tmp_path
    engine.dispose()


def headers(user=1):
    return {'Authorization':'Bearer '+create_access_token(user)}


def test_native_web_auth_and_owner_scoped_conversations(web):
    client,store,sessions,calls,path=web
    assert client.post('/api/chat/send',json={'content':'你好'}).status_code==401
    result=client.post('/api/chat/send',headers=headers(),json={'content':'我不喜欢咖啡','user_id':2})
    assert result.status_code==200
    chunks=[json.loads(line[6:]) for line in result.text.splitlines() if line.startswith('data: ')]
    assert chunks[-1]['done'] and ''.join(c['content'] for c in chunks)=='这是首次回答。'
    cid=chunks[-1]['conversation_id']
    assert client.post('/api/chat/send',headers=headers(2),json={'conversation_id':cid,'content':'你好'}).status_code==404
    assert store.records('web:1')[0]['polarity']=='negative' and store.records('web:2')==[]
    assert '旧库秘密饮品' not in json.dumps(calls,ensure_ascii=False)
    with sessions() as db:
        assert db.get(User,1).memory_note=='我喜欢旧库秘密饮品'
        assert db.query(Message).filter_by(conversation_id=cid,role='assistant').one().content=='这是首次回答。'
    trace=json.loads(next((path/'traces').glob('*.json')).read_text())
    assert trace['final']==trace['calls'][-1]['raw_first']=='这是首次回答。'


def test_web_memory_controls_use_new_store_only(web):
    client,store,sessions,_,_=web
    store.observe('web:1','fixture','我喜欢画画')
    store.observe('web:2','fixture','我喜欢游泳')
    assert client.get('/api/chat/memory',headers=headers()).json()['memory_note']=='我喜欢画画'
    assert client.put('/api/chat/memory',headers=headers(),json={'memory_note':'任意旧格式'}).status_code==422
    assert client.put('/api/chat/memory',headers=headers(),json={'memory_note':''}).status_code==200
    assert not store.records('web:1') and store.records('web:2')
    with sessions() as db:assert db.get(User,1).memory_note=='我喜欢旧库秘密饮品'


def test_new_conversation_model_reports_actual_runtime(web):
    client,*_=web
    result=client.post('/api/chat/conversations',headers=headers())
    assert result.status_code==200 and result.json()['model']=='test-double'
