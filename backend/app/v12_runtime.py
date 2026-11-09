"""Explicit in-process adapter; install only in an isolated verified 8B backend.

The existing login dependency and conversation database remain authoritative.
No client payload can choose a memory owner or a model. The runtime store must
be separate from the legacy memory_note column and other entrance stores.
"""
import asyncio
import json
import os
from pathlib import Path
import threading
import uuid

from atri_runtime import Runtime


class WebRuntime:
    def __init__(self, store, generator_factory, *, identity, search=None, search_current=None,
                 stage='daily', trace_dir=None):
        self.store, self.factory, self.identity = store, generator_factory, identity
        self.search, self.stage = search, stage
        self.search_current = search_current
        self.lock = threading.Lock()
        # Serialize the native web append/history/generate/persist transaction,
        # not just model.generate: another post must not become the current turn.
        self.request_lock = asyncio.Lock()
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True,exist_ok=True,mode=0o700)

    @staticmethod
    def owner(user_id):
        if type(user_id) is not int or user_id <= 0:
            raise ValueError('authenticated database user ID required')
        return 'web:'+str(user_id)

    def reply(self, user_id, message_id, messages):
        with self.lock:
            answer, trace = Runtime(self.store,self.factory({}),search=self.search,
                                   search_current=self.search_current,stage=self.stage).reply(
                self.owner(user_id),'web-message:'+str(message_id),messages)
            rid = uuid.uuid4().hex
            trace.update(request_id=rid,identity=self.identity,entrance='web')
            if self.trace_dir:
                fd=os.open(self.trace_dir/(rid+'.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'w') as f:json.dump(trace,f,ensure_ascii=False,indent=2)
            return answer, {'profile':trace['profile'],'request_id':rid,'operations':trace['operations']}

    def note(self,user_id):
        rows=self.store.records(self.owner(user_id))
        # Explicit subjects and original degree/negation remain visible.
        return '\n'.join(r['evidence'] for r in rows)

    def clear(self,user_id):
        with self.lock:
            return self.store.forget(self.owner(user_id))


def install(app, *args, **kwargs):
    if getattr(app.state,'v12_runtime',None) is not None:
        raise ValueError('V1.2 runtime already installed')
    app.state.v12_runtime=WebRuntime(*args,**kwargs)
    return app.state.v12_runtime
