from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atri_generation import hf_generator


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 9
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        return 'prompt'
    def __call__(self, prompt, **kwargs):
        return Batch(input_ids=torch.tensor([[1,2,3]]))
    def decode(self, ids, **kwargs):
        assert ids == [4,9]
        return '原始输出'


class Model:
    device = 'cpu'
    generation_config = SimpleNamespace(eos_token_id=9)
    def __init__(self):
        self.layer = SimpleNamespace(disable_adapters=False)
        self.modes = []
        self.fail = False
    def modules(self):
        return [self.layer]
    @contextmanager
    def disable_adapter(self):
        self.layer.disable_adapters = True
        try:
            yield
        finally:
            self.layer.disable_adapters = False
    def generate(self, **kwargs):
        self.modes.append(self.layer.disable_adapters)
        if self.fail:
            raise RuntimeError('generation failed')
        return torch.tensor([[1,2,3,4,9]])


def test_planner_uses_same_base_and_role_adapter_is_restored_for_answer():
    model = Model()
    generate = hf_generator(Tokenizer(), model, threading.Lock(), {'seed':17,'max_tokens':20})
    plan = generate([],kind='plan')
    answer = generate([],kind='answer')
    assert model.modes == [True,False]
    assert plan.metadata['adapter_enabled'] is False and answer.metadata['adapter_enabled'] is True
    assert answer == '原始输出' and answer.metadata['prompt_token_ids'] == [1,2,3]
    assert answer.metadata['output_token_ids'] == [4,9]
    assert answer.metadata['finish_reason'] == 'stop'


def test_failed_planning_cannot_leave_role_adapter_disabled():
    model = Model(); model.fail = True
    generate = hf_generator(Tokenizer(), model, threading.Lock(), {})
    with pytest.raises(RuntimeError):
        generate([],kind='plan')
    assert model.layer.disable_adapters is False


def test_unexpected_loaded_adapter_mode_is_rejected():
    model = Model(); model.layer.disable_adapters = True
    generate = hf_generator(Tokenizer(), model, threading.Lock(), {})
    with pytest.raises(ValueError,match='adapter state'):
        generate([],kind='answer')
    assert model.modes == []


def test_over_budget_input_is_rejected_before_generation_or_adapter_switch():
    class LongTokenizer(Tokenizer):
        def __call__(self,prompt,**kwargs):
            return Batch(input_ids=torch.ones((1,513),dtype=torch.int64))
    model=Model()
    generate=hf_generator(LongTokenizer(),model,threading.Lock(),{},max_input_tokens=512)
    with pytest.raises(ValueError,match='token budget'):
        generate([],kind='plan')
    assert model.modes==[] and model.layer.disable_adapters is False
