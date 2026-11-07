import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import atri_model_identity
from serve_memory_rag import file_sha
spec=importlib.util.spec_from_file_location('colab_v12',ROOT/'colab/v12_runtime.py')
colab=importlib.util.module_from_spec(spec);spec.loader.exec_module(colab)


def test_colab_mount_verifies_loaded_tensor_and_uses_authenticated_runtime(tmp_path,monkeypatch):
    base=tmp_path/'base';base.mkdir()
    adapter=tmp_path/'adapter';adapter.mkdir()
    config={'model_type':'qwen3','hidden_size':4,'num_hidden_layers':1,'num_attention_heads':1,'vocab_size':10}
    (base/'config.json').write_text(json.dumps(config))
    manifest=tmp_path/'manifest.json'
    digest=file_sha(base/'config.json')
    manifest.write_text(json.dumps({'base_files':{},'tokenizer_files':{'config.json':digest}}))
    # Tiny synthetic tensor fixture tests verification, not actual 8B identity.
    monkeypatch.setattr(atri_model_identity,'QWEN3_8B_CONFIG_SHA',digest)
    tensor=torch.tensor([[1.,2.]])
    save_file({'layer.lora_A.weight':tensor},adapter/'adapter_model.safetensors')
    model=SimpleNamespace(config=SimpleNamespace(**config,_name_or_path=str(base)),
                          named_parameters=lambda:[('layer.lora_A.default.weight',tensor)])
    tok=SimpleNamespace(name_or_path=str(base))
    kwargs=dict(tokenizer=tok,model=model,base_path=base,base_manifest=manifest,adapter_path=adapter,
        expected_adapter_sha=file_sha(adapter/'adapter_model.safetensors'),users={'fictional':'a'*40},
        database=tmp_path/'facts.db',system='你是亚托莉。',search=None,model_lock=threading.Lock())
    monkeypatch.setattr(colab,'hf_generator',lambda *a,**k: lambda messages,kind:
                        '{"utterance_kind":"self_report"}' if kind=='plan' else '首次输出')
    app=FastAPI();colab.mount_runtime(app,**kwargs)
    client=TestClient(app)
    body={'messages':[{'role':'user','content':'我喜欢画画'}],'user':'other'}
    assert client.post('/v12/v1/chat/completions',json=body).status_code==401
    headers={'Authorization':'Bearer '+'a'*40,'Idempotency-Key':'one'}
    result=client.post('/v12/v1/chat/completions',headers=headers,json=body)
    assert result.status_code==200 and result.json()['choices'][0]['message']['content']=='首次输出'
    assert client.get('/v12/v1/memory',headers=headers).json()['items'][0]['value']=='画画'
    tensor[0,0]=3
    with pytest.raises(ValueError,match='tensor differs'):
        colab.mount_runtime(FastAPI(),**kwargs)


def test_colab_requires_shared_lock_before_loading_any_state():
    with pytest.raises(ValueError,match='lock'):
        colab.mount_runtime(FastAPI(),tokenizer=None,model=None,base_path=None,base_manifest=None,
            adapter_path=None,expected_adapter_sha=None,users={},database=None,system='',search=None)
