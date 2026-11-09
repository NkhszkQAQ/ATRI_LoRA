# 主动 RAG 与回答安排：第一轮

隔离运行时可显式开启 `--rag-planning`，或在私有 JSON 配置中设置 `"rag_planning": true`，默认关闭。新增检索后的信息检查、最多两轮额外补查和带来源的回答安排。角色首次回答原样保留，训练、模型替换及部署另行决定。

## 调用与权限

首次 `plan` 及当前原话检索仍执行。之后 `review` 读取当前问题、近期对话、实际可用资料和实际操作结果；它只能请求 RAG 查询或选择资料 ID。每轮最多两个查询，最多两轮额外补查，**首次检索批次不计入这两轮**。重复规范化查询、没有新增准入资料、检索失败或两轮用尽后停止；允许再做一次信息检查形成最终交接，但不会继续检索。

`review` 使用同一底座并关闭角色 LoRA，`answer` 使用原 LoRA。可选规划 schema 仍只作用于首次 `plan`，不会将记忆提取格式错套到 review。review 的首次原始文本与校验结果均保留；格式或来源校验失败时保留缺口，不通过重答修复规划或角色首答。

补查只调用原来的 `StageCorpus.search`，沿用启动时的原作 SHA/场景范围校验、阶段内检索、既有重排阈值、跨查询合并排序与上下文预算。当前原话的可选 dense 补充权限不扩展到模型查询。服务阶段、身份、查询次数和预算不由模型或请求 body 决定。

自定义 search 回调属于受信程序边界，须满足与 StageCorpus 相同的来源核验合同；运行时不接受缺少证据的裸片段。模型不能提供回调或通过 JSON 自建来源。完整资料与交接共用原有 7000 字符预算，先为完整交接保留空间，再裁去排名靠后的整条资料；不截断已选事实。模型调用仍受原有 token 预算约束。

## 两层交接

`answer_plan` 由程序生成。review 只选择 node_id 和提出缺口；事实、出处、边界与操作结果由程序复制。

| 字段 | 含义与约束 |
| --- | --- |
| `must_respond` | 必须回应，每项含 source_id、完整 fact 与实际 evidence。只能选择送入 review 的资料，不接受模型自拟事实摘要。 |
| `optional_reference` | 相同的来源绑定格式，按需使用。未选择的预算内资料仍可作为普通 references。 |
| `fact_boundaries` | 按来源 ID 绑定完整命题。主体、否定、程度由完整事实保留，并附阶段、事件时间、揭示时间、知识状态。 |
| `information_gaps` | 仍未知、冲突及失败/停止提示；未命中不等于不存在。 |
| `operation_results` | 从实际执行 trace 复制并去除随机数据库 ID，不接受模型提供的成功状态。 |

另附 version、story_stage、stop_reason。信息足够与必须回应项的选择仍是模型的语义判断；ID 合法不证明核心事实齐全。角色提示要求自然回应，不朗读字段或查询过程；JSON/SSE 返回相同的首次角色回答，内部查询只在私有 trace 中保存。

本轮只新增 RAG 补查。既有记忆选择与操作位置保留，在补查之前仅执行一次；review 无写入/更正/删除入口。MEM 补查及完整操作失败验收属于下一阶段，不能因交接已有操作结果就宣称这些能力已通过。

## 验收与复现

程序测试覆盖两轮上限、重复查询、无新结果、首次/补充检索失败、来源 ID 伪造、阶段过滤、完整事实与共享预算、记忆操作不因补查重做、失败写入结果及 JSON/SSE 隔离。全套测试：595 passed、1 skipped、6 subtests passed；跳过项为未安装的可选规划格式库，保留 6 条既有依赖弃用警告。

使用旧日志的首次规划、旧角色首答及当前真实 CPU StageCorpus，完成 6 条旧题和 7 条预先封存表达变体的程序重放。学校、家务、未知接口及四种剧情阶段均覆盖，13/13 程序合同通过。学校与家务在补查后取得相应事实；未知接口未取得型号，停止并保留未知；剧情资料按阶段提供。未修改阈值、原作卡片、人设或角色权重。

**重放的 review 决策为明确标记的测试夹具，并非新模型输出。** 它证明给定查询/选择时程序能否执行正确，不证明模型会自主产生这些查询。旧角色首答只是不同旧输入下的历史证据，不是新输入下的 LoRA 验收结果。表达变体未用于调查询、阈值或语义实现。真实模型规划与角色首答全部未运行；按本轮用户选择未启动 GPU，不能据此定位 LoRA 训练对象，PR 保持 Draft。

`tools/eval_answer_planning.py` 支持互斥的 replay / real 模式。replay 显式传入历史目录与 review 夹具，记录来源哈希、实际命中、模型输入、回答安排、历史首答及虚构用户库前后状态。real 模式拒绝历史目录与 review 夹具，要求 `--execute`、底座 manifest、适配器 SHA 和显式本地路径；在空闲 GPU 核对加载身份后保存所有新鲜首次输出。评估器不读取最终留出集。

```sh
python tools/eval_answer_planning.py --mode replay \
  --cases /private/evaluation/cases.jsonl \
  --review-fixtures /private/evaluation/review-fixtures.json \
  --historical-dir /private/evaluation/old-results \
  --script-source /private/script-source \
  --reranker /private/models/frozen-reranker \
  --out /private/evaluation/new-replay
```

正式采用前需真实模型全链路复验：分别判断第二层的自主查询、补查、核心事实安排，以及第一层的完整回应、事实边界、未知表达和角色语气。网页 JWT 与 Colab 尚未增加实验开关；共享 Runtime、隔离 HTTP/QQ 构造器可显式启用，不表示真实 QQ 通讯或各平台验收通过。
