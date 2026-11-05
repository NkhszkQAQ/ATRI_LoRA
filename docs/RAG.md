# 本地剧本 RAG

当前 v1.3 服务候选版增加生成后的事实检查及来源回退，必须按 [v1.3 迁移说明](V1_3.md)重建索引。下方第二至第四轮保留各阶段结果，不能视为 v1.3 已通过真实 GPU 验收。

RAG 用小型中文 embedding 模型寻找相关剧本，再由原来的 Qwen3-8B + LoRA 回答。不重新训练 LoRA，也不替换推理服务。默认关闭，普通和 SSE 请求共用同一个注入位置。

本实现沿用 [LlamaIndex 本地教程](https://developers.llamaindex.ai/python/framework/getting_started/starter_example_local/) 的 `SimpleDirectoryReader`、`HuggingFaceEmbedding`、`VectorStoreIndex` 和默认磁盘存储。适配项目的部分：原作剧本先解析 JSON 场景；embedding 使用已确认的中文 `BAAI/bge-base-zh-v1.5`；检索通过 `as_retriever()` 得到原始片段，直接注入已有 LoRA 服务。没有安装 Ollama 集成或 AgentWorkflow，不要求 ATRI 模型调用工具，也不使用另外一个模型先写摘要。

## 安装

建议 Python 3.12，RAG 依赖与训练依赖分开：

```bash
python3.12 -m venv .venv-rag
source .venv-rag/bin/activate
pip install -r misc/requirements-rag.txt
```

首次构建或加载会从 Hugging Face 下载 embedding 权重。可用 `HF_HOME` 将缓存放在数据盘；缓存完整后可设置 `HF_HUB_OFFLINE=1`。向量化使用 CPU，不占用 LoRA 的 CUDA 显存。实际 8B 推理仍需要原来的 CUDA、Transformers、PEFT、bitsandbytes、FastAPI、uvicorn 环境；RAG requirements 不能代替这些推理依赖。

## 构建与检查索引

只选择明确用于知识库的目录。首批使用原作剧本，不把训练集、聊天记录或整个归档作为 source。

```bash
export HF_HOME="$PWD/.rag-cache/huggingface"
export TIKTOKEN_CACHE_DIR="$PWD/.rag-cache/tiktoken"
# 部分 Transformers 版本会为旧权重启动额外的后台格式转换下载。
export DISABLE_SAFETENSORS_CONVERSION=1
python tools/atri_rag.py build \
  --source /Users/nkhszk/atri/_共用资料/原作剧本 \
  --storage ./rag_storage-v1.3 \
  --persona ./persona \
  --facts ./rag/character_facts.json

python tools/atri_rag.py search --storage ./rag_storage-v1.3 '亚托莉是谁？'
```

支持 UTF-8 `.md`、普通 `.txt`，以及 JSON 数组格式的剧本（`.json` 或以 JSON 开头的 `.txt`）。剧本按场景生成 Document，保留说话者与从 1 开始的对白编号，再由 SentenceSplitter 切分。不跟随目录内的符号链接，跳过隐藏路径。JSON 格式不符合剧本结构时明确失败，不默默丢弃资料。开头为 `[` 或 `{` 的文本按 JSON 校验，普通资料如有这种开头请先整理成 Markdown 正文。

输出的每个命中包含 node_id、相似度、正文，以及 source（相对路径）、文件 SHA256、scene（从 1 开始）、scene_type、entry_count、start_char/end_char。字符位置指向提取后的场景文本，而非原 JSON 字节位置。分支选项单独标注为“分支选项（不代表已经发生）”，不把选项当作剧情事实。可按 source + SHA256 + scene 找回原始资料；节点标识在已保存索引中稳定，重建后可能变化。

`atri_manifest.json` 保存文件清单、内容哈希、文档数、节点数及 embedding/切分配置。现有目录不会被构建命令覆盖；资料变化后建立新目录，检查检索结果，再切换 `ATRI_RAG_STORAGE`。运行时读取的是索引快照，不自动扫描原文件，也不自动检测资料是否更新。索引含原文，按私有资料保存，不上传 GitHub。

## 接入现有服务

在已能正常运行 8B LoRA 的环境中安装 RAG 依赖，将 `tools/atri_rag.py`、`tools/atri_grounding.py`、`tools/atri_memory.py` 和 `tools/serve_v1_1_8b.py` 放在同一目录。使用原有的底座和 adapter，并使用本版 system 文件：

```bash
export ATRI_BASE=/path/to/Qwen3-8B
export ATRI_ADAPTER=/path/to/adapter
export ATRI_SYSTEM=/path/to/atri-system.txt
export ATRI_RAG_ENABLED=1
export ATRI_RAG_STORAGE=/path/to/rag_storage-v1.3
python tools/serve_v1_1_8b.py
```

开启后，在创建 FastAPI 应用时加载索引一次。没有 storage、索引损坏或模型/切分配置不匹配时启动失败并报错。关闭时不导入 LlamaIndex 或加载 embedding；设 `ATRI_RAG_ENABLED=0` 并重启即可。v1.3 入口先检查配置和索引，再加载 LoRA。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `ATRI_RAG_ENABLED` | `0` | `1`、`true`、`yes` 开启 |
| `ATRI_RAG_STORAGE` | 无 | 开启时必须指定的索引目录 |
| `ATRI_RAG_EMBED_MODEL` | `BAAI/bge-base-zh-v1.5` | 建索引与查询必须相同 |
| `ATRI_RAG_CHUNK_SIZE` | `256` | LlamaIndex tokenizer 的切分 token 数 |
| `ATRI_RAG_CHUNK_OVERLAP` | `32` | 相邻片段重叠 token 数 |
| `ATRI_RAG_TOP_K` | `3` | 最终最大检索片段数，先召回至少 12 个候选 |
| `ATRI_RAG_MIN_SCORE` | `0.55` | 关闭重排时的向量相似度过滤阈值 |
| `ATRI_RAG_REWRITE` | `1` | 启用有界规则化查询改写 |
| `ATRI_RAG_RERANK` | `1` | 启用本地交叉编码器重排；RAG 总开关仍默认关闭 |
| `ATRI_RAG_RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 单输出多语言重排模型 |
| `ATRI_RAG_RERANK_MIN_SCORE` | `0.5` | sigmoid 归一化后的重排过滤阈值，不是校准概率 |
| `ATRI_RAG_CANDIDATES` | `20` | 每个查询的通用候选数，另召回最多10个事实卡片段 |
| `ATRI_RAG_MAX_CONTEXT_CHARS` | `4000` | 注入的参考资料 JSON 字符预算 |

embedding 的输入上限为 512 个自身 tokenizer token。切分 tokenizer 与 embedding tokenizer 不同，极端长文本可能被 embedding 截断；切分参数需结合命中质量调整，调整后重建索引。字符预算只限制新增参考资料，不限制用户历史；超长会话仍由现有客户端管理。

每轮用最后一条 user 消息检索。历史、人设、原始问题均保留，资料 JSON 只在本轮推理中注入，不改数据库。空检索结果沿用原消息。召回后先按相关性阈值过滤，对已核对的事实卡给予 0.04 的小幅排序加分，再截取前 k 项。固定问候直接跳过检索。命中仍不代表事实正确，生成引用也不保证必然遵循来源。资料作为参考数据处理，不能保证完全消除提示注入。

普通与流式请求格式不变，可继续通过 AstrBot 等 OpenAI 兼容客户端使用；接口不新增公开 sources 字段，需追溯来源时用 search 命令检查。RAG 不自动接入其他 4B、Colab 或完整版 backend 入口。

## 测试

```bash
pip install pytest fastapi httpx
python -m pytest tests/test_rag.py -q
```

测试包含脚本转换、来源、私有路径排除、索引落盘/重载、配置不匹配与覆盖保护、上下文预算、不改原消息，以及 RAG 开/关下普通和 SSE 接口。自动化检索测试使用确定性测试 embedding；接口测试使用替代模型，不能代替真实中文 embedding 的质量检查或 GPU LoRA 验证。

实际部署前，在 GPU 环境确认 RAG 开/关均能回答、SSE 正常结束、原有 seed/采样参数保持一致，再用多条已知答案的问题人工核对命中来源与生成内容。此步骤不由本地单元测试替代。

## 本次本地验证（2026-11-05）

- Python 3.12.14，llama-index-core 0.14.25、embeddings-huggingface 0.6.1、readers-file 0.5.6。
- 自动化测试 13 项通过；普通与 SSE 请求使用替代生成模型，未加载真实 LoRA。
- 真实 `BAAI/bge-base-zh-v1.5` 加载成功，输出 768 维向量。完整 34 个剧本导入 1,069 个场景（含 3 个分支场景），生成 2,237 个节点，并成功从磁盘重新加载检索。
- 三条人工抽查问题中，“高性能”问题检索到对应自称片段；夏生腿部问题的前 3 项中包含右腿疼痛片段，但首项偏向“作为腿陪伴”的比喻。“亚托莉是谁”未命中足以直接说明身份的资料，宽泛问题的检索质量仍需校准。这里只确认真实检索链路运行成功，不作为检索准确率评估。
- 本机未做 CUDA/NF4 + PEFT LoRA 的真实生成、真实 GPU SSE、QQ 收发或部署验证。合并或部署前应补齐这些推理相关检查并评估检索质量。


## 补充资料与检索调校（第二轮，历史结果）

`--persona` 只导入指定目录中的 persona_card.md 和 character_profile.md，标记为 project_persona。导入时将旧卡“不吃东西”的错误替换为可以进食但不能吸收营养的原作结论，记录 normalization，原文件保持不变。其他称呼、篇幅、陪伴互动规则作为项目改写资料，不能当作原作设定。当前系统提示词仍是角色控制的最高来源，本次不修改它；若旧 system 本身包含错误，真实生成能否正确区分仍需验证。

`--facts` 导入 rag/character_facts.json 的 10 张概括卡，标记 canonical_fact。每张卡在建索引前校验原作文件 SHA256、场景与对白范围；不匹配则报错。事实卡保留早期剧情范围，不将序章海底相遇断言为人生中第一次见面，也不凭机器人身份编造充电方式。卡片保留出处指针，不包含原作大段引文。

长期记忆 PR #11 已合入本任务分支。记忆认证及明确命令处理保持原有行为；普通聊天先对原始问题进行 RAG，再注入对应用户的记忆 JSON，避免让记忆文本成为检索问题。共同开启、各自关闭的普通/SSE 请求均有测试覆盖，记忆命令绕过检索与生成。

新版完整索引为 1,081 个 Document / 2,281 个节点：1,069 场景 + 2 份项目资料 + 10 张事实卡。索引只保存在本地。自动化测试共 39 项通过，其中包括上游长期记忆测试；生成仍使用替代模型。

38 条固定回归用例分为 19 条校准输入和 19 条预先准备的改写输入，每组 12 个事实检索问题与 7 个应不注入问题。校准只比较 0.50、0.55、0.60、0.65、0.70 五个阈值；0.50/0.55 均 19/19，选较保守的 0.55。固定参数后，改写组为 15/19。失败是“她是一个真正的人类女孩吗”“你尝得出食物的味道吗”“她是不是很会做饭和料理”“战斗型号是已经证实的身份吗”；本轮未据此继续修改参数。这是针对已整理事实的小型回归集，不是独立语料上的通用准确率评测。命中条件为正确事实卡在前 3 项，不衡量最终回答正确率，也不保证其余片段均相关。

可复现命令：

```bash
python tools/eval_rag.py --storage ./rag_storage-v2 --calibrate --output ./calibration.json
python tools/eval_rag.py --storage ./rag_storage-v2 --output ./evaluation.json
```

结果包含剧本片段，应放在私有目录，避免提交 Git。索引配置与命中详情只供本地审阅。

剩余问题主要是代词/省略实体时的检索召回。已查阅官方的 [Query Transformations](https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/query_transformations/) 与 [Rerankers](https://developers.llamaindex.ai/python/framework/module_guides/models/rerankers/)：可以将追问结合上下文改写为自足问题，或更广召回后通过交叉编码器重排。本轮未额外引入这些模型、框架或模型调用；需要另行验证其延迟、资源消耗、拒绝无关资料的效果与多人物指代歧义。GPU LoRA、真实 GPU SSE 和 QQ 链路仍未在本任务验证，PR 保留 Draft。


## 第三轮：规则化改写与本地重排

本轮沿用相同 v2 索引、事实卡和原38题，没有针对失败问题添加答案。实现参考官方 [查询变换](https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/query_transformations/) 和 [重排](https://developers.llamaindex.ai/python/framework/module_guides/models/rerankers/)，通过已安装的 Sentence Transformers CrossEncoder 执行重排，无需新增框架或调用外部生成API。采用多语言 [mMiniLMv2 重排模型](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)，CPU执行；首次使用另下载约470MB权重到 HF_HOME 缓存。启用 RAG 时，两项增强默认开启；不需要重建索引。

流程为原问题及最多一个改写问题分别召回，合并去重，再使用改写后的完整问题对候选重排。普通向量召回每个查询最多20项，另按 kind=canonical_fact 过滤召回最多10个事实卡片段，防止概括卡被大量对白挤出候选池。重排开启时用0.35作为较宽的初始向量门槛，之后按0.5的归一化重排分数过滤，沿用事实卡0.04的排序加分，最终最多3项。返回 score 此时是重排相关性分数，同时返回 retrieval_score、retrieval_query 和 rerank_query 以追溯。分数不是事实真实性概率。

改写保留原句，只补全检索问题：直接对助手说“你”时补成亚托莉；独立问题以该角色语料库的亚托莉为默认主体；第三人称参考最近一条用户消息中明确的已知人物。如果该条消息出现多个人物，或第三人称指向未识别的人物，则返回空检索，避免擅自把“她”套成亚托莉。识别人名采用有限列表，并非通用指代消解。助手历史不用于选主体，/memory 指令也不用于选主体。烹饪能力问题补全为家务类能力问法，避免重排模型把做饭误匹配为进食；这是有限领域规则，不是通用LLM改写或答案生成。个人日常状态陈述、固定问候跳过资料检索。多意图、陌生名字和其他复杂指代仍需更广泛评估。

增强逻辑只读取当前请求的原始对话，既不读取也不写入长期记忆数据库。服务器继续先认证/处理记忆命令，再做 RAG，最后注入用户自己的长期记忆。调用者的人设、原始问题和历史对象保持不变。重排在检索锁内执行，防止并发共享模型的中间数据串用；结果去重及分数保存在本次调用中，不写入索引节点。关闭 RAG 仍可独立运行服务；关闭重排不加载第二模型。模型缺失、配置错误不会静默降级为不经过滤的检索。

测试结果：55项自动化测试通过，包含上游长期记忆原测试、普通/SSE各种开关组合、用户隔离、未授权请求、记忆命令绕过RAG，以及改写边界、重排分数归一化/过滤和来源保留。真实模型下，原38题38/38通过，新增16题16/16、多轮边界4题4/4通过。原来的4个失败问题首项分别是 identity、taste、housework、housework 事实卡。改写组现已用于针对性开发，是回归集，不能再当作独立未知测试集；新增问题也不是大规模准确率基准。判定仍是正确事实卡位于前3项或不适用资料时返回空，真实生成正确率没有据此得到验证。

重排器并不总能理解问题：中途测试将“会做饭”误匹配到进食卡，单靠重排只修复3个问题。最终通过能力问法的类目补全及单独召回事实卡解决，因此不能推断任意新问法均已覆盖。GPU LoRA、真实GPU SSE、QQ端到端以及现有人设与知识资料的最终生成一致性仍未在本任务验证，PR继续保留Draft。

复现第三轮及新增问题：

```bash
export HF_HOME="$PWD/.rag-cache/huggingface"
export DISABLE_SAFETENSORS_CONVERSION=1
# 权重完整缓存后才设置离线模式
export HF_HUB_OFFLINE=1
python tools/eval_rag.py --storage ./rag_storage-v2 --output .rag-cache/evaluation-v3.json
python tools/eval_rag.py --storage ./rag_storage-v2 --cases tests/fixtures/rag_eval_v3_extra.json --output .rag-cache/evaluation-v3-extra.json
```

`--calibrate` 在重排开启时仅比较校准分组的重排阈值，关闭重排时比较向量阈值。若希望暂停增强，设 ATRI_RAG_RERANK=0 和/或 ATRI_RAG_REWRITE=0 并重启；索引与长期记忆不受开关影响。


## 第四轮：真实8B LoRA与长期记忆组合测试（2026-11-05）

本轮不修改运行时代码，在空闲RTX 5090的独立环境中实际加载Qwen3-8B + v1.1-R adapter，NF4/bfloat16计算；adapter SHA256与发布权重一致。Torch 2.12.1+cu130、Transformers 5.6.0、PEFT 0.18.1、bitsandbytes 0.50.2。真实embedding/重排权重及重排词表哈希与本地一致。使用当前仓库公开system模板，仅展开占位符；两组均启用临时长期记忆，10题分别比较RAG开/关，主seed=20261104、temperature=0.7、top_p=0.9、max_tokens=320。共37条真实生成，另含独立进程重启和最小诊断。

实际回答说明检索回归通过并不等于生成正确：

| 观察项 | 真实结果 |
| --- | --- |
| 味觉/进食 | RAG关闭时说尝不出、不能像人一样吃饭；开启后能明确能尝味道、能进食但不能补充营养 |
| 身份/腿部/序章相遇 | 主要事实改善，但仍有无依据称呼、角色名字误写和范围表达问题 |
| 料理 | 正确检索housework卡后仍答成已经学会做饭；当前生成未通过 |
| 学校动机 | 只说学习，漏掉寻找记忆，还否认其他原因 |
| 未核实的充电设定 | 检索为空，模型仍泛泛编造而非明确承认未知 |
| 两用户记忆 | 单独提问能读出各自保存的偏好与代号；一条混合回答把用户偏好写成了ATRI自身偏好 |
| SSE | 训练机127.0.0.1真实HTTP/SSE正常完成并收到[DONE]，但混合问题有漏答；传输完成不是回答完整 |

15项后端/传输检查全部通过，含保存、应用重建、真正独立进程重启保留、双用户数据库隔离、无效认证拒绝、忘记/清空及HTTP/SSE。复核最终输入只含相应用户的记忆，未发现测试A/B记忆串库；生成中的主语混淆不能等同于已发现数据库泄露。普通请求经同一FastAPI ASGI处理器执行真实CUDA生成，另外用临时uvicorn验证本机HTTP/SSE；未做真实QQ或外网客户端验证，也没有高并发压力测试。

最小诊断中，完整上下文料理题3个seed仍有误导说法；临时只放事实卡后2个seed明确否定家务能力，另1个仍含糊，提示资料范围混杂和模型忠实性均需处理。加强记忆归属的临时system提示改善了1次混合回答，但充电题仍未明确承认未知。诊断只存在于测试进程，不是已提交或验证充分的修复。

因此撤销“真实GPU尚未执行”这一历史验证缺口，同时以已观察到的生成忠实性、主语归属、多问题完整性和称呼问题作为新的明确阻塞项，PR继续Draft。原38/58题记录仍只是检索回归，不能当作实际回答准确率。未发布、未合并本PR。

已获授权上传原作索引至训练机隔离目录。取回结果后已删除远端测试目录（索引、临时SQLite/访问密钥、环境和模型缓存），原训练环境与adapter保留、adapter哈希未变，GPU回到0 MiB。完整原始实答保存在开发者本地私有测试记录，未上传用户记忆或私人提示词到本仓库。
