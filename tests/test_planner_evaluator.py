"""Diagnostic execution must not misidentify a role model or stale source copy."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def evaluator():
    path = Path(__file__).resolve().parents[1] / 'tools/eval_planner_only.py'
    spec = importlib.util.spec_from_file_location('planner_evaluator_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('case', ['adapter', 'wrong_quantization'])
def test_diagnostic_rejects_model_mismatch_before_creating_output(tmp_path, case):
    model = SimpleNamespace(is_loaded_in_4bit=True, config=SimpleNamespace(
        model_type='qwen3', quantization_config={'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': 'float16',
            'bnb_4bit_use_double_quant': False}))
    if case == 'adapter':
        model.peft_config = {'default': object()}
    out = tmp_path / 'results'
    with pytest.raises(ValueError):
        evaluator().evaluate(None, model, root=tmp_path, script_source=tmp_path,
                             cases_path=tmp_path / 'cases.jsonl', out=out, reranker=None)
    assert not out.exists()


def test_frozen_source_import_does_not_reuse_other_notebook_version(tmp_path):
    module = evaluator()
    versions = []
    for number in (1, 2):
        root = tmp_path / str(number)
        package = root / 'src/atri_core'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        (package / 'probe.py').write_text(f'VERSION = {number}\n')
        core, seal = module.modules(root)
        versions.append((core('probe').VERSION, core('probe').__name__, seal))
    assert [row[0] for row in versions] == [1, 2]
    assert versions[0][1] != versions[1][1] and versions[0][2] != versions[1][2]


def test_failed_generation_preserves_actual_request_without_mock_answer(tmp_path, monkeypatch):
    import torch
    module = evaluator()
    root = Path(__file__).resolve().parents[1]
    core, _ = module.modules(root)
    def fail(*args, **kwargs):
        raise RuntimeError('simulated model failure')
    monkeypatch.setattr(core('inference'), 'hf_generator', lambda *args, **kwargs: fail)
    monkeypatch.setattr(core('rag'), 'StageCorpus', lambda *args, **kwargs:
        SimpleNamespace(search=lambda *args: [], search_current=lambda *args: []))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'unit-test: no GPU')
    model = SimpleNamespace(is_loaded_in_4bit=True, config=SimpleNamespace(
        model_type='qwen3', quantization_config={'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': 'bfloat16',
            'bnb_4bit_use_double_quant': False}),
        generation_config=SimpleNamespace(to_dict=lambda: {}), eval=lambda: None)
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'failure', 'stage': 'daily', 'user': '测试委托'}) + '\n')
    out = tmp_path / 'results'
    with pytest.raises(RuntimeError, match='simulated model failure'):
        module.evaluate(None, model, root=root, script_source=tmp_path,
                        cases_path=cases, out=out, reranker=None)
    request = json.loads((out / 'requests.jsonl').read_text())
    assert request['kind'] == 'plan' and request['request_id'] == 0
    assert '测试委托' in request['messages'][-1]['content']
    assert not (out / 'raw-model.jsonl').exists()
    assert json.loads((out / 'FAILURE.json').read_text())['model_calls'] == 0


@pytest.mark.parametrize('expect_needs', [False, True])
def test_no_lookup_passes_only_when_case_explicitly_expects_no_facts(tmp_path, monkeypatch, expect_needs):
    import torch
    module = evaluator()
    root = Path(__file__).resolve().parents[1]
    core, _ = module.modules(root)
    outputs = {'plan': '{"utterance_kind":"other"}', 'lookup_plan': 'null',
               'review': '{"source_ids":[],"gap":"","queries":[],"optional_reference":[]}'}
    monkeypatch.setattr(core('inference'), 'hf_generator', lambda *args, **kwargs:
                        lambda messages, *, kind: core('inference').GeneratedText(outputs[kind], {}))
    monkeypatch.setattr(core('rag'), 'StageCorpus', lambda *args, **kwargs: SimpleNamespace(
        search=lambda *args: [], search_current=lambda *args: [], source_hashes={}))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'unit-test: no GPU')
    model = SimpleNamespace(is_loaded_in_4bit=True, config=SimpleNamespace(
        model_type='qwen3', quantization_config={'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': 'bfloat16',
            'bnb_4bit_use_double_quant': False}),
        generation_config=SimpleNamespace(to_dict=lambda: {}), eval=lambda: None)
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'control', 'group': 'unit_test', 'stage': 'daily',
        'user': '你以前的主人是谁？' if expect_needs else '你好',
        'expected_required': [], 'expect_needs': expect_needs}) + '\n')
    result = module.evaluate(None, model, root=root, script_source=tmp_path,
        cases_path=cases, out=tmp_path / 'results', reranker=None)['results'][0]
    assert result['program_checks']['needs_match_expectation'] is (not expect_needs)
    assert result['program_checks_pass'] is (not expect_needs)
