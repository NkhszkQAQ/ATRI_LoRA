"""Optional token-level planning format; semantic authorization stays in Runtime."""
import hashlib
from importlib.metadata import distribution
import json
import threading
from functools import lru_cache


def _object(properties, required=None):
    return {'type': 'object', 'properties': properties,
            'required': list(properties) if required is None else required,
            'additionalProperties': False}


def _string(**kwargs):
    return {'type': 'string', **kwargs}


PLAN_SCHEMA = _object({
    'utterance_kind': _string(enum=['self_report', 'quotation', 'joke', 'temporary', 'other']),
    'memory_scores': {'type': 'array', 'maxItems': 128, 'items': _object({
        'id': _string(minLength=1, maxLength=80),
        'relevance': {'type': 'number', 'minimum': 0, 'maximum': 1}})},
    'queries': {'type': 'array', 'maxItems': 2,
                'items': _string(minLength=1, maxLength=180)},
    'candidates': {'type': 'array', 'maxItems': 16, 'items': _object({
        'subject': _string(minLength=1, maxLength=160),
        'predicate': _string(enum=['preference', 'name', 'allergy']),
        'value': _string(minLength=1, maxLength=256),
        'evidence': _string(minLength=1, maxLength=1024),
        'polarity': _string(enum=['positive', 'negative']),
        'degree': _string(enum=['ordinary', 'favorite']),
        'supersedes': _string(minLength=1, maxLength=80)},
        required=['subject', 'predicate', 'value', 'evidence', 'polarity', 'degree'])},
    'forget_ids': {'type': 'array', 'maxItems': 16,
                   'items': _string(minLength=1, maxLength=80)},
})
SCHEMA_SHA256 = hashlib.sha256(json.dumps(PLAN_SCHEMA, sort_keys=True).encode()).hexdigest()
_lock = threading.Lock()


@lru_cache(maxsize=2)
def _tokenizer_data(tokenizer):
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    return build_token_enforcer_tokenizer_data(tokenizer)


def planner_constraint(tokenizer):
    """Cache vocabulary only; create a fresh parser for every request.

    This cannot verify evidence, owner, intent, or numerical confidence. The
    runtime still rejects unsafe candidates and IDs. Truncation remains a
    recorded failure; there is no repair or second generation here.
    """
    from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    with _lock:
        data = _tokenizer_data(tokenizer)
    # The parser caches its alphabet for minLength strings at construction.
    # Supplying the tokenizer alphabet only through TokenEnforcer is too late
    # for Chinese evidence/value strings in the pinned library.
    parser = JsonSchemaParser(PLAN_SCHEMA,
        config=CharacterLevelParserConfig(alphabet=data.tokenizer_alphabet))
    callback = build_transformers_prefix_allowed_tokens_fn(data, parser)
    package = distribution('lm-format-enforcer')
    source = json.loads(package.read_text('direct_url.json') or '{}')
    return callback, {'library': 'lm-format-enforcer', 'version': package.version,
                      'source_commit': source.get('vcs_info', {}).get('commit_id'),
                      'schema_sha256': SCHEMA_SHA256}
