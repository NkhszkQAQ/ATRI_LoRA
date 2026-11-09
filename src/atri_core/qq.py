"""Signed current-event adapter for isolated QQ private-chat V1.2 requests.

Identity and event ID originate in AstrBot, not in model output or body.user.
The existing V1 signature is retained for compatibility with PR #17.
"""
import base64
import hashlib
import hmac
import json
import re
import time

from .bridge import detach_context, identity

EVENT = re.compile(r'\n<ATRI_EVENT_V12>([A-Za-z0-9_=-]+)\.([0-9a-f]{64})</ATRI_EVENT_V12>')


def attach_event(system, event, prompt, secret, *, now=None):
    owner = identity(event)
    message = getattr(event, 'message_obj', None)
    event_id = str(getattr(message, 'message_id', '') or '')
    if not event_id or len(event_id) > 160 or not isinstance(prompt, str):
        raise ValueError('verified current QQ message ID and text required')
    current = time.time() if now is None else now
    data = {'owner': owner, 'private': bool(event.is_private_chat()), 'event_id': event_id,
            'time': current, 'text_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    encoded = base64.urlsafe_b64encode(json.dumps(data, ensure_ascii=False, separators=(',',':')).encode()).decode()
    signature = hmac.new(secret, encoded.encode(), hashlib.sha256).hexdigest()
    return EVENT.sub('', system or '') + f'\n<ATRI_EVENT_V12>{encoded}.{signature}</ATRI_EVENT_V12>'


def resolve_event(body, secret, *, context_window=None, now=None):
    from fastapi import HTTPException
    try:
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages or any(not isinstance(m,dict) for m in messages):
            raise ValueError('messages required')
        messages, context = detach_context(messages, secret, now=now)
        if context is None or not context['private']:
            raise ValueError('signed private QQ context required')
        frames = []
        for m in messages:
            if m.get('role') == 'system' and isinstance(m.get('content'),str):
                matches = list(EVENT.finditer(m['content']))
                if m['content'].count('<ATRI_EVENT_V12>') != len(matches):
                    raise ValueError('malformed event signature')
                frames.extend(match.groups() for match in matches)
                m['content'] = EVENT.sub('', m['content'])
        if len(frames) != 1:
            raise ValueError('one current event signature required')
        encoded, signature = frames[0]
        if len(encoded)>4096 or not hmac.compare_digest(signature,hmac.new(secret,encoded.encode(),hashlib.sha256).hexdigest()):
            raise ValueError('invalid event signature')
        event = json.loads(base64.urlsafe_b64decode(encoded))
        age = (time.time() if now is None else now)-event['time']
        if not -10 <= age <= 180 or event['owner'] != context['owner'] or event['private'] is not True:
            raise ValueError('event scope or time mismatch')
        if not isinstance(event['event_id'],str) or not 0 < len(event['event_id']) <= 160:
            raise ValueError('invalid event ID')
        last = messages[-1]
        if last.get('role') != 'user' or not isinstance(last.get('content'),str):
            raise ValueError('text event required')
        if hashlib.sha256(last['content'].encode()).hexdigest() != event['text_sha256']:
            raise ValueError('current user text is not signed')
        if any(m.get('role') not in {'system','user','assistant'} or not isinstance(m.get('content'),str) for m in messages):
            raise ValueError('V1.2 QQ adapter accepts text conversations only')
        notice = ''
        if context_window is not None:
            before = next((m['content'] for m in messages if m['role']=='system'),'')
            messages, _ = context_window.select(messages, context, now=now)
            after = next((m['content'] for m in messages if m['role']=='system'),'')
            if after.startswith(before):
                notice = after[len(before):]
        history = [{'role':m['role'],'content':m['content']} for m in messages if m.get('role') in {'user','assistant'}]
        return event['owner'], 'qq-event:'+event['event_id'], history, notice
    except (ValueError,TypeError,KeyError,UnicodeError) as exc:
        raise HTTPException(401,'valid signed private QQ event required') from exc


def build_qq_app(store, secret, generator_factory, *, system, identity, search=None, search_current=None,
                 stage='daily', trace_dir=None, context_window=None, rag_planning=False):
    from .http import build_app
    if not isinstance(secret,bytes) or len(secret)<32:
        raise ValueError('bridge secret must be at least 32 bytes')
    return build_app(store, {}, generator_factory, system=system, identity=identity, search=search,
                     search_current=search_current, stage=stage, trace_dir=trace_dir,
                     rag_planning=rag_planning,
                     principal_resolver=lambda body: resolve_event(body,secret,context_window=context_window))
