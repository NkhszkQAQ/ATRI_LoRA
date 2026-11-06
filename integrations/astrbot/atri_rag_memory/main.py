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
        rows = await asyncio.to_thread(self.store.list, owner) if private else []
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
