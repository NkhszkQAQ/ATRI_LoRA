# 运维速查

## 网络与服务

- Docker 网络：`atri-companion-net`（仅本项目）
- 服务：`postgres` · `ollama`（复用 `ollama/ollama:0.17.7`）· `api` · `web`
- 默认端口（仅绑定 127.0.0.1）：Web `3080` · API `3081` · Ollama `11435` · Postgres `5433`

## 初始化

```bash
cp .env.example .env
# 修改 SECRET_KEY、POSTGRES_PASSWORD、ADMIN_*、可选 SMTP_*

docker compose up -d --build

docker compose run --rm -e ADMIN_PASSWORD='你的强密码' api \
  python -m app.cli init --yes --admin-email admin@example.com --admin-password "$ADMIN_PASSWORD"
```

交互式（可现场配置 SMTP）：

```bash
docker compose run --rm -it api python -m app.cli init
```

## 模型自动选择

`docker/ollama/select_model.sh`：

1. `nvidia-smi` 读显存  
2. ≤10GB → `qwen3:4b`；更大 → `qwen3:8b`  
3. 已有则复用，没有则 `ollama pull`  
4. 有 TTY 时等待 `MODEL_CONFIRM_SECONDS`（默认 10）可手动输入标签  
5. 无 TTY（默认 compose）直接自动确认  

强制指定：`.env` 里 `MODEL_FORCE=qwen3:4b`

管理后台 **底层模型** 页可再次覆盖（写入 DB `system_settings`）。

## 安全能力

| 能力 | 行为 |
|------|------|
| 套取提示词（中/英/日/韩/俄/西/法/德等） | 拦截并角色化拒绝，记 `safety_events` |
| 自伤/自杀倾向 | 危机回复 + 记事件（高/危） |
| 情绪失控 | 记事件，对话可继续 |
| 管理员查看原文/聊天 | 强制写 `audit_logs` |

## 常用命令

```bash
docker compose logs -f ollama
docker compose logs -f api
docker compose exec ollama ollama list
docker compose down
```
