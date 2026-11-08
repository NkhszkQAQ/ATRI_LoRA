"""Check evaluator routing without claiming real-model or hosted-platform acceptance."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from atri_memory_lifecycle import LifecycleStore
from atri_qq_v12 import build_qq_app
from eval_v12_entrances import check_qq_group_isolation,make_web,validate_development_review


def test_review_requires_same_retrieval_configuration():
    review={'ready_for_entrance_validation':True,'adapter_sha256':'adapter','dataset_sha256':'cases'}
    args={'adapter_sha':'adapter','dataset_sha':'cases'}
    disabled={'model':None,'threshold':None}
    enabled={'model':'/isolated/bge','threshold':.4}
    validate_development_review(review,**args,story_dense=disabled)
    with pytest.raises(ValueError,match='retrieval configuration'):
        validate_development_review(review,**args,story_dense=enabled)
    review['story_dense']=enabled
    validate_development_review(review,**args,story_dense=enabled)
    with pytest.raises(ValueError,match='retrieval configuration'):
        validate_development_review(review,**args,story_dense=disabled)


def test_partial_dense_cli_configuration_fails_before_loading_files(tmp_path):
    command=[sys.executable,str(ROOT/'tools/eval_v12_entrances.py')]
    for name in ('base','adapter','base-manifest','system','seal','cases','script-source','out','development-review'):
        command.extend(['--'+name,str(tmp_path/name)])
    command.extend(['--adapter-sha','a'*64,'--story-dense-model',str(tmp_path/'bge')])
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'dense model and calibrated threshold' in result.stderr
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('stream',[False,True])
def test_signed_group_probe_is_rejected_before_model_access(tmp_path,stream):
    store=LifecycleStore(tmp_path/'memory.sqlite3')
    secret=b'fictional-entrance-evaluator-test-secret'
    def factory(body):
        pytest.fail('group traffic reached the model factory')
    app=build_qq_app(store,secret,factory,system='你是亚托莉',identity={'model':'test-double'})
    with TestClient(app) as client:
        evidence=check_qq_group_isolation(client,store,secret,stream=stream)
    assert evidence['http_status']==401 and evidence['private_records_unchanged']
    assert store.records('qq:isolated:900000001')[0]['value']=='荔枝'


def test_native_web_evaluator_passes_current_retrieval_into_model_input(tmp_path,monkeypatch):
    # make_web changes only test environment variables; restore after this check.
    for name in ('DATABASE_URL','SECRET_KEY','SYSTEM_PROMPT_PATH'):
        monkeypatch.setenv(name,'unused-test-value')
    queries=[];answers=[]
    def current(query,history):
        queries.append(query)
        return [{'node_id':'canonical:entrance-fixture','text':'测试资料：可以到露台看看。',
                 'score':.9,'metadata':{'allowed_stages':['daily'],'kind':'canonical_summary',
                                      'source':'facts/entrance-fixture'}}]
    def factory(options):
        def generate(messages,*,kind):
            if kind=='plan':return '{}'
            answers.append(messages)
            return '首次测试输出'
        return generate
    store=LifecycleStore(tmp_path/'memory.sqlite3')
    web=make_web(ROOT,tmp_path,store,factory,{'model':'test-double'},lambda q,h:[],
                 ROOT/'persona/system_prompt.txt',1,search_current=current)
    try:
        with TestClient(web['app']) as client:
            result=client.post('/api/chat/send',headers=web['headers'](1),json={'content':'我们去散步吧'})
            assert result.status_code==200
            chunks=[json.loads(line[6:]) for line in result.text.splitlines() if line.startswith('data: ')]
            assert chunks[-1]['done'] and ''.join(c['content'] for c in chunks)=='首次测试输出'
        assert queries==['我们去散步吧'] and len(answers)==1
        assert 'canonical:entrance-fixture' in answers[0][-1]['content']
    finally:
        web['engine'].dispose()
