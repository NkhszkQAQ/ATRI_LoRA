import json
from pathlib import Path
from types import SimpleNamespace
import sys

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from atri_bridge import attach_context
from atri_memory_lifecycle import LifecycleStore
from atri_qq_v12 import attach_event, resolve_event, build_qq_app

SECRET=b'fictional-isolated-test-secret-12345678'


def body(sender='111',text='我喜欢画画',private=True,now=None):
    event=SimpleNamespace(message_obj=SimpleNamespace(message_id='event1'),
        get_platform_id=lambda:'test',get_sender_id=lambda:sender,is_private_chat=lambda:private)
    system=attach_context('untrusted external persona','qq:test:'+sender,[],private,SECRET,now=now)
    system=attach_event(system,event,text,SECRET,now=now)
    return {'messages':[{'role':'system','content':system},{'role':'user','content':text}]}


def test_current_event_text_identity_and_id_are_bound():
    request=body(now=100)
    owner,event,history,notice=resolve_event(request,SECRET,now=101)
    assert (owner,event)==('qq:test:111','qq-event:event1')
    assert history==[{'role':'user','content':'我喜欢画画'}]
    request['messages'][-1]['content']='我喜欢唱歌'
    with pytest.raises(HTTPException):resolve_event(request,SECRET,now=101)
    with pytest.raises(HTTPException):resolve_event(body(now=100),SECRET,now=400)


def test_text_part_history_and_current_turn_are_normalized_without_mutating_archive():
    request=body(now=100)
    request['messages'][0]['content']=[{'type':'text','text':request['messages'][0]['content']}]
    request['messages'][1:1]=[
        {'role':'user','content':[{'type':'text','text':'旧的'},{'type':'text','text':'问题'}]},
        {'role':'assistant','content':'旧的回答'}]
    request['messages'][-1]['content']=[{'type':'text','text':'我喜欢'},{'type':'text','text':'画画'}]
    original=json.loads(json.dumps(request))
    owner,event,history,_=resolve_event(request,SECRET,now=101)
    assert (owner,event)==('qq:test:111','qq-event:event1')
    assert history==[{'role':'user','content':'旧的问题'},
                     {'role':'assistant','content':'旧的回答'},
                     {'role':'user','content':'我喜欢画画'}]
    assert request==original
    request['messages'][-1]['content'][-1]['text']='唱歌'
    with pytest.raises(HTTPException):resolve_event(request,SECRET,now=101)


@pytest.mark.parametrize('parts',[
    [{'type':'image_url','image_url':{'url':'https://example.invalid/image'}}],
    [{'type':'text','text':'我喜欢画画'},{'type':'input_audio','input_audio':{}}],
    [{'type':'text','text':None}],['我喜欢画画'],
    [{'type':'text','text':'我喜欢画画','image_url':{}}],
])
def test_non_text_parts_are_rejected_without_silently_dropping_them(parts):
    request=body(now=100)
    request['messages'][-1]['content']=parts
    with pytest.raises(HTTPException):resolve_event(request,SECRET,now=101)


def test_group_unsigned_and_wrong_secret_cannot_access_memory():
    for request,secret in [(body(private=False),SECRET),({'messages':[{'role':'user','content':'你好'}]},SECRET),
                           (body(),b'x'*40)]:
        with pytest.raises(HTTPException):resolve_event(request,secret)


def test_signed_qq_transport_uses_shared_runtime_and_ignores_body_owner(tmp_path):
    store=LifecycleStore(tmp_path/'facts.db')
    seen=[]
    def factory(options):
        def generate(messages,*,kind):
            seen.append(messages)
            return '{"utterance_kind":"self_report"}' if kind=='plan' else '原始回复'
        return generate
    app=build_qq_app(store,SECRET,factory,system='你是亚托莉。',identity={'model':'test-double'})
    client=TestClient(app)
    request={**body(),'owner':'qq:test:222','user':'qq:test:222','stream':True}
    request['messages'][-1]['content']=[{'type':'text','text':'我喜欢画画'}]
    result=client.post('/v1/chat/completions',json=request)
    assert result.status_code==200 and '[DONE]' in result.text
    assert store.records('qq:test:111')[0]['value']=='画画'
    assert store.records('qq:test:222')==[]
    assert 'untrusted external persona' not in json.dumps(seen)
    client.post('/v1/chat/completions',json=request)
    assert store.records('qq:test:111')[0]['confirmations']==1
    assert client.get('/v1/memory').status_code==403


def test_context_selector_clipping_and_generated_time_notice_are_kept():
    class Context:
        def select(self,messages,context,now=None):
            assert context['owner']=='qq:test:111'
            selected=[{**messages[0],'content':messages[0]['content']+'\nTrusted timestamp'},messages[-1]]
            return selected,{}
    request=body()
    request['messages'][1:1]=[{'role':'user','content':'old'},{'role':'assistant','content':'old reply'}]
    _,_,history,notice=resolve_event(request,SECRET,context_window=Context())
    assert len(history)==1 and notice=='\nTrusted timestamp'
