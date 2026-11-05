# 本地剧本 RAG

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
  --storage ./rag_storage-v2 \
  --persona ./persona \
  --facts ./rag/character_facts.json

python tools/atri_rag.py search --storage ./rag_storage-v2 '亚托莉是谁？'
```

支持 UTF-8 `.md`、普通 `.txt`，以及 JSON 数组格式的剧本（`.json` 或以 JSON 开头的 `.txt`）。剧本按场景生成 Document，保留说话者与从 1 开始的对白编号，再由 SentenceSplitter 切分。不跟随目录内的符号链接，跳过隐藏路径。JSON 格式不符合剧本结构时明确失败，不默默丢弃资料。开头为 `[` 或 `{` 的文本按 JSON 校验，普通资料如有这种开头请先整理成 Markdown 正文。

输出的每个命中包含 node_id、相似度、正文，以及 source（相对路径）、文件 SHA256、scene（从 1 开始）、scene_type、entry_count、start_char/end_char。字符位置指向提取后的场景文本，而非原 JSON 字节位置。分支选项单独标注为“分支选项（不代表已经发生）”，不把选项当作剧情事实。可按 source + SHA256 + scene 找回原始资料；节点标识在已保存索引中稳定，重建后可能变化。

`atri_manifest.json` 保存文件清单、内容哈希、文档数、节点数及 embedding/切分配置。现有目录不会被构建命令覆盖；资料变化后建立新目录，检查检索结果，再切换 `ATRI_RAG_STORAGE`。运行时读取的是索引快照，不自动扫描原文件，也不自动检测资料是否更新。索引含原文，按私有资料保存，不上传 GitHub。

## 接入现有服务

在已能正常运行 8B LoRA 的环境中安装 RAG 依赖，将 `tools/atri_rag.py` 和 `tools/serve_v1_1_8b.py` 放在同一目录。使用原有的底座、adapter 和 system 路径：

```bash
export ATRI_BASE=/path/to/Qwen3-8B
export ATRI_ADAPTER=/path/to/adapter
export ATRI_SYSTEM=/path/to/atri-system.txt
export ATRI_RAG_ENABLED=1
export ATRI_RAG_STORAGE=/path/to/rag_storage-v2
python tools/serve_v1_1_8b.py
```

开启后，在创建 FastAPI 应用时加载索引一次。没有 storage、索引损坏或模型/切分配置不匹配时启动失败并报错。关闭时不导入 LlamaIndex 或加载 embedding；设 `ATRI_RAG_ENABLED=0` 并重启即可。当前入口先加载 LoRA 再创建应用，RAG 启动错误不会提前避免 LoRA 的加载耗时。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `ATRI_RAG_ENABLED` | `0` | `1`、`true`、`yes` 开启 |
| `ATRI_RAG_STORAGE` | 无 | 开启时必须指定的索引目录 |
| `ATRI_RAG_EMBED_MODEL` | `BAAI/bge-base-zh-v1.5` | 建索引与查询必须相同 |
| `ATRI_RAG_CHUNK_SIZE` | `256` | LlamaIndex tokenizer 的切分 token 数 |
| `ATRI_RAG_CHUNK_OVERLAP` | `32` | 相邻片段重叠 token 数 |
| `ATRI_RAG_TOP_K` | `3` | 最终最大检索片段数，先召回至少 12 个候选 |
| `ATRI_RAG_MIN_SCORE` | `0.55` | 相似度过滤阈值，在本项目小型校准集上选取，不代表通用可信度 |
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


## 补充资料与检索调校（第二轮）

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
