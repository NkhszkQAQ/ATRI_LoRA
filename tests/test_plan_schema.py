import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
lmfe = pytest.importorskip('lmformatenforcer')
from atri_plan_schema import PLAN_SCHEMA, planner_constraint


def accepted(value):
    text = json.dumps(value, ensure_ascii=False)
    # TokenEnforcer sets this from the tokenizer; direct character tests must
    # supply their alphabet too (the library default contains ASCII only).
    config = lmfe.CharacterLevelParserConfig()
    config.alphabet += text
    parser = lmfe.JsonSchemaParser(PLAN_SCHEMA, config=config)
    for char in text:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


def plan():
    return dict(utterance_kind='self_report', memory_scores=[], queries=[], forget_ids=[],
                candidates=[dict(subject='user', predicate='preference', value='蓝色',
                    evidence='我把喜欢的颜色改成蓝色。', polarity='positive', degree='ordinary',
                    supersedes='m1')])


def test_correction_requires_evidence_and_valid_enums_before_completion():
    item = plan()
    assert accepted(item)
    del item['candidates'][0]['evidence']
    assert not accepted(item)
    item = plan(); item['candidates'][0]['predicate'] = 'like'
    assert not accepted(item)
    item = plan(); item['utterance_kind'] = 'unknown'
    assert not accepted(item)


def test_empty_plan_is_legal_without_forcing_a_save_or_query():
    item = plan(); item['candidates'] = []; item['utterance_kind'] = 'other'
    assert accepted(item)


def test_statement_shape_is_supported_and_runtime_enforces_numeric_bounds():
    from atri_runtime import model_candidates
    item = plan()
    item['candidates'] = [dict(subject='user', predicate='statement', value='我长期需要拐杖',
        evidence='我长期需要拐杖', polarity='positive', degree='ordinary', importance=.9, stability=1)]
    assert accepted(item)
    item['candidates'][0]['importance'] = 1.2
    # The pinned format parser does not enforce numeric minimum/maximum.
    # Runtime admission must reject the value regardless of token filtering.
    assert model_candidates(item['candidates'], '我长期需要拐杖') == []
    assert accepted(plan())


def test_schema_is_not_a_semantic_or_authorization_check():
    # Well-formed but false/foreign evidence must still pass through Runtime's
    # existing evidence and ownership checks, never directly into persistence.
    item = plan(); item['candidates'][0]['supersedes'] = 'another-owner-id'
    item['candidates'][0]['evidence'] = '模型编造的证据'
    assert accepted(item)


def test_transformers_prefix_filters_tokens_with_independent_request_state():
    import torch
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    chars = sorted(set('0' + json.dumps(plan(), ensure_ascii=False)))
    vocab = {'[UNK]': 0, '[EOS]': 1, **{c: i+2 for i, c in enumerate(chars)}}
    backend = Tokenizer(WordLevel(vocab, '[UNK]'))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                                       unk_token='[UNK]', eos_token='[EOS]')
    callback, meta = planner_constraint(tokenizer)
    other, _ = planner_constraint(tokenizer)
    prefix = [vocab['0']]
    assert vocab['{'] in callback(0, torch.tensor(prefix))
    assert vocab['}'] not in callback(0, torch.tensor(prefix))
    assert tokenizer.eos_token_id not in callback(0, torch.tensor(prefix))
    callback(0, torch.tensor(prefix + [vocab['{']]))
    assert vocab['{'] in other(0, torch.tensor(prefix))
    for char in json.dumps(plan(), ensure_ascii=False):
        assert vocab[char] in other(0, torch.tensor(prefix))
        prefix.append(vocab[char])
    assert tokenizer.eos_token_id in other(0, torch.tensor(prefix))
    assert len(meta['schema_sha256']) == 64
