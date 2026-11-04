# 跨会话记忆

## 原则

记忆**不靠 LoRA 权重**，由服务端维护并注入 system：

```text
User.memory_note  →  load_system_prompt() 替换 {{memory}}  →  模型当轮可见
```

## 写入

| 方式 | 接口 |
|------|------|
| 自动抽取 | 用户发消息时 `update_user_memory()`（喜好/疲劳/睡眠等启发式） |
| 用户手动 | `GET/PUT /api/chat/memory` |
| 管理员 | `PATCH /api/admin/users/{id}` 的 `memory_note`；或 `GET /api/admin/users/by-name/{name}/memory` |

条目上限约 12 条 / 1200 字，同类前缀会覆盖更新。

## 训练对齐

`data/v2/scenarios/with_memory.jsonl`：system 里带真实记忆块，教模型**自然引用、不复读、不编造**。  
`meta.ctx=with_memory` 仅作元信息；只有推理时也会提供的内容才写入 system。

## 后续可升级

- 会话结束用小模型做摘要再合并  
- 向量检索长期笔记  
- 用户「忘记这件事」指令  
