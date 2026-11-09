"""Private-chat explicit memory; QQ identity comes from AstrBot events only."""
import asyncio
import json
import math
from pathlib import Path
import time

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.star.filter.command import GreedyStr

from .atri_bridge import attach_context, identity, read_secret
from .atri_memory import MemoryStore, command_reply


class Main(star.Star):
    def __init__(self, context: star.Context):
        super().__init__(context)
        settings = json.loads(Path(__file__).with_name("settings.json").read_text())
        config = json.loads(Path(settings["config_path"]).read_text())
        self.secret = read_secret(config["shared_secret_file"])
        self.lifecycle = None
        if config.get('runtime_profile') == 'memory-rag-v1.2-dev':
            from .atri_memory_lifecycle import LifecycleStore
            self.lifecycle = LifecycleStore(config['lifecycle_db'])
            self.store = None
        else:
            self.store = MemoryStore(config["memory_db"])

    @filter.command("memory")
    async def memory(self, event: AstrMessageEvent, rest: GreedyStr):
        if event.get_platform_name() != "aiocqhttp":
            reply = "此记忆入口用于 QQ 私聊。"
        elif not event.is_private_chat():
            reply = "长期记忆只在私聊中保存、查看和删除，请私聊使用 /memory。"
        else:
            try:
                owner = identity(event)
                command = "/memory" + (" " + rest.strip() if rest.strip() else "")
                if self.lifecycle is not None:
                    if rest.strip() in {'', 'list'}:
                        rows = await asyncio.to_thread(self.lifecycle.records, owner)
                        reply = '\n'.join(r['id']+'：'+r['evidence'] for r in rows) or '尚无长期记忆。'
                    elif rest.strip().startswith('del '):
                        event_id = str(getattr(getattr(event,'message_obj',None),'message_id','') or '')
                        if not event_id:
                            raise ValueError('缺少当前消息 ID')
                        result = await asyncio.to_thread(self.lifecycle.forget_event,owner,
                            'qq-command:'+event_id,command,[rest.strip()[4:].strip()])
                        reply = '已删除。' if result['status']=='deleted' else '这条删除请求已经处理过。'
                    else:
                        reply = '用 /memory list 查看，/memory del 记录ID 删除；保存或更正可以直接在对话中说明。'
                else:
                    raw = await asyncio.to_thread(self.store.command, owner, command)
                    reply = command_reply(command, raw)
            except ValueError as exc:
                reply = "未保存：" + str(exc)
        yield event.plain_result(reply)
        event.stop_event()

    @filter.on_llm_request()
    async def memory_context(self, event: AstrMessageEvent, req: ProviderRequest):
        if event.get_platform_name() != "aiocqhttp":
            return
        owner = identity(event)
        private = event.is_private_chat()
        rows = await asyncio.to_thread(self.store.list, owner) if private and self.store is not None else []
        turn = None
        conversation = getattr(req, "conversation", None)
        if private and conversation and getattr(conversation, "cid", None):
            now = time.time()
            sent = getattr(getattr(event, "message_obj", None), "timestamp", None)
            if type(sent) not in (int, float) or not math.isfinite(sent) or not 0 < sent <= now + 10:
                sent = now
            previous = getattr(conversation, "updated_at", 0)
            if type(previous) not in (int, float) or not math.isfinite(previous) or previous < 0:
                previous = 0
            turn = {"conversation_id": str(conversation.cid), "sent_at": sent, "previous_at": previous}
        req.system_prompt = attach_context(req.system_prompt, owner, rows, private, self.secret, turn=turn)
        if self.lifecycle is not None:
            from .atri_qq_v12 import attach_event
            req.system_prompt = attach_event(req.system_prompt,event,req.prompt,self.secret)
