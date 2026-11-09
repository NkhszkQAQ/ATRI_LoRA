# 共享核心整合与影响说明

本次是源码和配置整合，不是模型训练或上线。模型能力的 V1.2 验收仍未通过；现有权重、实验记录、原作文件和生产数据库不迁移。

## 日常入口

公共代码以本仓库为准。模型与原作资料留在独立档案目录，私有实验、配置、密钥和数据库留在私有目录，旧开发仓库保留历史和未提交资产。Git worktree 是同一仓库的任务目录，不是另一套主项目。

```bash
# 只安装本仓库代码，不自动下载模型、安装训练栈或启动服务
python -m pip install --no-deps -e .

# 查看统一服务入口，旧的长参数启动方式仍然兼容
python tools/serve_memory_rag.py --help
# 填写好私有配置后才手动启动；示例文件本身不能直接运行
atri-memory-rag --config /private/path/runtime.json
```

`configs/runtime.example.json` 是统一服务的配置模板。相对路径以配置文件所在目录为准；复制到私有位置后应修改路径。HTTP/Colab 的 Bearer 用户映射与 QQ 的签名密钥二选一。QQ 使用 `qq_secret` 代替 `users`，可显式指定独立的 `qq_context_store`。所有入口继续隔离所有者、事件和数据库。

## 实现归属

| 模块 | 唯一实现位置 | 保留的旧导入入口 |
| --- | --- | --- |
| 调用顺序 | `src/atri_core/runtime.py` | `tools/atri_runtime.py` |
| 规划与候选校验 | `src/atri_core/planner.py` | 从 `atri_runtime` 导出相同函数 |
| 生命周期与事务 | `src/atri_core/memory.py` | `tools/atri_memory_lifecycle.py` |
| 来源、阶段与检索准入 | `src/atri_core/rag.py` | `tools/atri_story.py` |
| 索引、embedding、重排工具 | `src/atri_core/retrieval.py` | `tools/atri_rag.py` |
| 人设展开及规划/回答规则 | `src/atri_core/prompt.py` | 旧 8B 服务导出同一个展开函数 |
| 模型加载与生成适配 | `src/atri_core/inference.py` | `tools/atri_generation.py`、旧 8B 的 `load()` |
| 权重和加载身份 | `src/atri_core/identity.py`、`files.py` | `tools/atri_model_identity.py` |
| 短期上下文 | `src/atri_core/context.py` | `tools/atri_context.py` |
| HTTP、认证与 QQ 签名 | `src/atri_core/http.py`、`auth.py`、`bridge.py`、`qq.py` | 原服务和 QQ 导入名 |
| 显式记忆和事实回退兼容模式 | `src/atri_core/legacy/` | 原 `atri_memory`、`atri_grounding` |

旧导入是模块别名，指向同一份实现，不复制另一套可变模型或存储状态。核心不再从 `tools/serve_memory_rag.py` 反向导入哈希工具或模型加载器。模型依赖按需加载，单纯导入核心不会开模型或创建数据库。

## 四入口和流程

HTTP 由共享服务直接提供。QQ 从 AstrBot 事件认证身份，再交给同一服务；网页保留 JWT 和会话事务，通过 `backend/app/v12_runtime.py` 显式安装共享运行时；Colab 在已有模型上调用 `colab/v12_runtime.py` 并传入同一模型锁。它们使用同一生命周期、检索和回答实现。

顺序保持为：认证身份/事件 → 时间上下文 → 规划 → 校验并执行记忆事务 → 阶段检索和预算 → 组装人设与实际资料/操作结果 → 角色第一次回答 → 返回结果与独立 trace。模型候选不能选择数据库所有者；失败操作不会被程序伪装成成功。

规划默认关闭角色 LoRA，回答启用角色 LoRA。共享代码没有改变采样、训练目标、人设原文或事实卡。原始模型错误仍会真实保留，不能把整合当成模型质量改善。

QQ 文字传输同时兼容字符串和 OpenAI 的纯 `text` 内容分段列表，包含 AstrBot 已存的用户消息格式。认证前仅规范化请求副本；历史库不被改写，当前文字仍需匹配消息签名。图片、音频、混合内容及被篡改的文字继续拒绝，不会静默丢弃后伪装成纯文字请求。

PR #17 的上下文实现和签名 `turn` 字段已按来源吸收，保留 PR #18 的当前事件签名。指定 QQ 上下文库时才在新生命周期入口应用它；旧 QQ 代理也保留同样的时间窗口。没有合并或关闭 PR #17。

## QQ 插件交付变化

整合后不能只复制两个 `tools/*.py` 文件安装插件，它们现在是兼容入口。使用源码打包器生成自包含目录：

```bash
python tools/package_astrbot_plugin.py --output /private/path/new-atri-plugin
```

输出必须是新目录；包含插件、兼容入口和 `src/atri_core`，附逐文件 SHA 清单。不会包含设置、密钥、数据库或权重，也不会安装或重启 AstrBot。之后由部署者在受限目录配置 `settings.json`。已经安装的旧插件不会被本地整合自动改变。

## 保留和归档边界

旧显式记忆与事实回退实现已集中到 `src/atri_core/legacy`，原导入名保持兼容。它们仍服务旧接口，不再作为新版自动记忆的实现入口。

网页版 `User.memory_note`、旧 Mac CLI、匿名 Colab 页面、4B/DT 和旧多模型网关仍是兼容入口。它们尚未全部具备新版语义验收，不能直接删除或把现有用户数据迁入生命周期库。旧匿名 Colab 页面不会因为这次整合自动获得个人长期记忆。后续切换这些入口，需要各自显式配置和验证。

所有模型候选、训练配置、原始首答、失败复盘、旧索引和部署证据均保留。历史归档不是自动训练数据源。不能批量导入 `.bak`、实验目录或真实聊天。

## 验证与限制

程序测试使用虚构身份、临时 SQLite 和替代生成器。冻结的整合前回放同时检查完整模型输入、记忆操作、数据库内容、阶段过滤和首次回答，避免只核对函数名。QQ 插件另在不加载 checkout/site-packages 的独立进程验证自包含导入；旧命令的帮助入口不需要 GPU。

真实 GPU、QQ 通讯、Google Colab 平台和 Docker 镜像构建没有在本轮执行。Docker 构建上下文和 COPY/挂载路径已按当前目录修复，仍需有 Docker 的环境进行实际 Build。正式模型质量、全部入口和最终留出门槛继续保留，PR 维持 Draft。

## 撤回本次源码整合

本次没有数据库迁移或线上切换。需要回到旧源码时，可使用 Git 中整合前的提交创建独立 checkout；不要覆盖本地未提交工作。新配置和打包目录可保留；模型、实验和记忆数据无需恢复或重写。

## 本轮实际验证记录

- 在独立临时测试环境中，默认 `pytest`：576 项通过、6 个子检查通过；可选 `lmformatenforcer` 模块因未安装跳过 1 个测试模块。保留 6 条既有依赖弃用警告。
- 整合前冻结回放与整合后结果一致，包括完整模型输入、首次回答、SQLite 操作、用户隔离、引用保护及阶段过滤。这里使用替代生成器，不是新运行的真实 8B 验收。
- Python wheel 构建通过，独立加载 wheel 中的核心和服务帮助入口通过；QQ 自包含包在移除 site-packages 的子进程中可导入并使用虚构记忆库。
- Docker COPY 源路径检查通过；没有 Docker 环境，未进行镜像 Build。
- 原人设、事实卡、原 1.1-R 权重，以及主副本中原有两处未提交修改的哈希保持一致。未修改模型档案、私有实验或生产数据。

修复两项已有测试夹具问题：入口替身补齐 `search_current`；并发容量测试更新实际已保存的键，不假定任一固定线程必定赢得名额。测试发现并修正了网页 Docker 的可选 lockfile COPY 问题；源码未新增前端布局或交互变化。
