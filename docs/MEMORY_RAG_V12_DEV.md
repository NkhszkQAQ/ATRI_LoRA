# 自动记忆与分阶段原作检索：开发配置

状态：实现与程序测试进行中，**尚未通过真实 8B 验收**。本文件描述可配置的开发入口，不宣称发布新权重。能力目标是 V1.2；运行配置名为 `memory-rag-v1.2-dev`，不改现有程序 1.3.0 或远端模型标签。

## 数据与职责

| 层 | 负责内容 |
| --- | --- |
| `atri_memory_lifecycle` | SQLite 所属用户隔离、当前事件去重、原子保存/更正/删除、重要性/确认时间/重复确认评分 |
| `atri_runtime` | 同一 8B 分析上下文、选择记忆、提出查询和候选；程序检查证据并执行，再把实际结果交给角色回答 |
| `atri_story` | 核对原作文件 SHA 与场景出处，先按角色实际知识阶段过滤，再用本地交叉编码器重排 |
| 各入口 | 从 Bearer、网页 JWT 或 QQ 签名取得身份；分别保留会话及数据库，不接受模型或 body.user 指定所属用户 |
| `atri_generation` | 同一底座内部规划默认临时禁用角色 LoRA；最终回答启用原有 1.1-R。锁内恢复适配器，保存首次输出及 token |

默认保存门槛为 `.8`，另要求稳定性至少 `.7`；入库评分为 `.5×明确度 + .35×稳定性 + .15×重要性`。明确一次的重要事实可以保存。当前自动提取类型是偏好、名字和过敏信息，不代表支持任意事实。引用、玩笑、临时选择、助手建议及不确定分析不会入库。严格证据检查可能漏存复杂省略表达，必须通过真实测试确认覆盖。

取回先过 `.35` 相关性门槛，再按 `.65×相关性 + .2×重要性 + .15×近期` 排序。近期为 `2^(-距用户确认天数/(30×强度))`，强度为 `min(4,1+.5×(独立用户确认次数-1))`。参数来自 `Policy`，可用 `--memory-policy` JSON 配置。它们是工程起点，不是人脑定律；参考 [Generative Agents §4.1](https://arxiv.org/html/2304.03442v2) 和 [MemoryBank §2.3](https://arxiv.org/html/2305.10250v1)。

读取、助手复述和请求重试不增加确认次数。不同偏好可以并存；同对象的明确否定/程度更正令旧版本失效，指定更正目标必须同用户同主体同类型。删除清除相应历史版本，与事件摘要同事务；旧保存/删除事件重放不会恢复或误删后来事实。降权不会自动删除，包括明确要求保留的重要信息。

原作摘要、来源场景、揭示阶段和角色知识与个人事实分开。原作没有个人遗忘规则，也不会被模型写入。`daily`、`school_later`、`eden_partial`、`past_recovered` 由服务端配置，客户端不能自行切阶段。

## 独立 HTTP

从仓库根目录运行，所有路径必须明确指定，使用新的隔离数据库及虚构用户令牌文件：

```sh
python tools/serve_memory_rag.py \
  --base /isolated/Qwen3-8B --adapter /isolated/1.1-R \
  --adapter-sha EXPECTED_SHA256 --base-manifest /isolated/base-manifest.json \
  --system persona/system_prompt.txt --users /isolated/users.json \
  --database /isolated/http-memory.sqlite3 --traces /isolated/http-traces \
  --script-source /isolated/original-scripts \
  --story-cards rag/character_facts.json rag/story_stages.json rag/story_daily_extended.json \
  --stage daily --port 8022
```

底座 manifest 包含 `base_files` 和 `tokenizer_files` 两个文件名到 SHA256 的映射。启动校验底座、分词器、适配器文件，以及实际加载的每个 LoRA 张量。`/health` 返回加载身份，不代表语义验收合格。

`POST /v1/chat/completions` 使用每用户 Bearer，并必须带稳定的 `Idempotency-Key`。仅接受文本 `user/assistant` 消息，系统人设固定在服务端；这只是 OpenAI 风格接口的受限子集。支持 JSON 和缓冲 SSE，两者展示同一个首次回答。没有修复生成或固定答案兜底。`GET /v1/memory` 仅返回当前用户条目，`DELETE /v1/memory?value=...` 删除指定值，省略值显式删除全部。不要把 API 令牌放进公共 notebook。

## QQ 私聊入口

启动时用 `--qq-secret /isolated/bridge-secret` 替代 `--users`。只接受带有效旧身份签名及新当前事件签名的私聊；当前文本、所属用户、消息 ID 和时间必须吻合。群聊、未签名请求、改写后的文本以及媒体消息均拒绝，不读取私人事实。

隔离 AstrBot 插件沿用 `integrations/astrbot/atri_rag_memory/main.py`，放入同目录的工具副本为 `atri_bridge.py`、`atri_memory.py`、`atri_memory_lifecycle.py`、`atri_qq_v12.py`。在新的运行配置中设置 `runtime_profile: memory-rag-v1.2-dev`、`lifecycle_db` 和 `shared_secret_file`；插件与服务的生命周期数据库必须映射到同一个新文件。**不要指向旧 memory_db。** 不配置此 profile 时保留旧行为。

`/memory list` 显示本人事实及 ID，`/memory del ID` 删除；自然对话负责自动保存/更正。消息 ID 和请求 hook 使用 [AstrBot 官方接口](https://github.com/AstrBotDevs/AstrBot/blob/master/docs/en/dev/star/guides/listen-message-event.md)。

PR #17 提供的 `atri_context.ContextWindow` 可通过 `--qq-context-store` 连接。独立分支未复制或合并 #17；已在其当前提交代码上验证签名时间元数据及跨日裁剪的程序兼容性。合并时插件 hook 的两处改动需要保留：先附带 #17 的 `turn`，随后签名当前事件。尚未宣称真实 QQ 通讯或生产机器人验收。

## 网页与 Colab

网页入口使用 `backend/app/v12_runtime.install(app, store, generator_factory, identity=..., search=..., stage=..., trace_dir=...)` 显式安装到隔离后端。`PYTHONPATH` 须包含仓库 `tools` 和 `backend`；模型与后端需在同一进程，工厂复用 `atri_generation.hf_generator` 和模型锁。本配置暂不提供 Ollama 远程转接。原 JWT 用户和会话权限继续生效，SSE 形状保持兼容；新模式不读写旧 `User.memory_note`。记忆页面可查看新事实、清空全部；非空文本批量覆盖返回 422，保存/更正通过对话完成。未安装时保留旧后端路径。

Colab 在已有模型加载后调用 `colab/v12_runtime.mount_runtime(...)`，明确传入底座、校验 manifest、适配器及 SHA、虚构用户令牌、新数据库、检索器和原 notebook 共用的模型锁。它再次核对文件和已加载 LoRA 张量，再挂载 `/v12/v1/chat/completions`。原匿名演示页不获得共享记忆身份，未改写旧 notebook。尚未进行 Google Colab 平台实测。

## 验证与当前限制

`tools/eval_memory_rag.py` 对比原服务链路和改进链路，固定同一底座/1.1-R/非思考模式/采样参数，并记录差异。原链路的重试、修复和兜底与原始输出一并保留。改进链路额外使用底座规划调用、阶段过滤摘要和生命周期操作，不能把整个差异归因于权重。

开发/最终留出题、完成门槛在首次模型结果前封存于独立验收目录；最终留出需显式 `--allow-final`，不放进运行配置或训练。完整开发对照已运行 264 个 HTTP 轮次（36 个场景、3 个种子、两条链路），真实结果仍存在主体、程度、未知天气、保存失败虚报及漏检索错误，未达标。一次隔离 SFT+KL 候选也未通过 132 次相同输入对照，未采用；角色权重仍为原 1.1-R。最终留出和各入口真实模型批次尚未运行。程序测试成功不表示 V1.2 达标、PR 可合并或已经部署。

上述开发结果对应较早程序快照；后续已修复过敏否定、候选证据句末标点，并让检索始终使用当前用户话和近期用户上下文，模型查询作为补充。原作示例问题仅用于检索排序，不再传入回答参考。当前程序修改尚需真实 8B 复验，不能把旧快照结果当成这些修复的验证。

多查询命中现在先按资料 ID 合并、保留最高相关分，再排序分配上下文预算，避免较弱的先到结果挤掉后续高相关资料。本人候选使用明确陈述句式校验，不能仅因以“我”开头就把“我的室友喜欢……”存成用户喜好；明确的关系主体仍可按独立主体保存。复杂省略表达的覆盖仍需实测。

### 可选的规划格式约束实验

安装 `misc/requirements-v12-planner.txt` 后，独立服务及对照评估器可显式加 `--planner-schema`。它用 [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer) 的 token 过滤要求规划器输出必要字段和合法枚举，默认关闭。固定的上游提交包含 Transformers 5 导入兼容修复；PyPI 0.11.3 在当前环境无法导入该集成。

约束只作用于内部规划，最终聊天仍为不受格式约束的首次生成。trace 记录库版本和 schema SHA；省略、截断或语义错误不经重试修复。合法 JSON 不代表事实真实、主体正确或删除获授权，现有逐字证据和用户隔离检查仍执行。该选项尚未经过真实 8B 对照，不用于宣称上述失败已解决。
