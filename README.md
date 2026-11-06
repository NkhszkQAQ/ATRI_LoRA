<div align="center">

# ATRI LoRA

**「我是高性能的嘛！」**

基于《ATRI -My Dear Moments-》角色亚托莉的 Qwen3 陪伴聊天 LoRA

权重 · 训练配置 · 推理服务 · Colab 一键体验

[![Stars](https://img.shields.io/github/stars/NkhszkQAQ/ATRI_LoRA?style=flat-square&logo=github&color=f4b8c5)](https://github.com/NkhszkQAQ/ATRI_LoRA/stargazers)
[![Release](https://img.shields.io/github/v/release/NkhszkQAQ/ATRI_LoRA?style=flat-square&color=89c4e1&label=latest)](https://github.com/NkhszkQAQ/ATRI_LoRA/releases)
[![License](https://img.shields.io/badge/code-Apache--2.0-a5d6a7?style=flat-square)](LICENSE)
[![Base](https://img.shields.io/badge/base-Qwen3--4B%20%7C%208B-b39ddb?style=flat-square)](https://huggingface.co/Qwen)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-f9ab00?style=flat-square&logo=googlecolab&logoColor=white)](colab/ATRI_LoRA_Colab.ipynb)

[快速开始](#-快速开始) · [版本选择](#-版本选择) · [训练](#-自己训练) · [推理服务](#-推理服务) · [长期记忆](#-长期记忆) · [已知限制](#-已知限制) · [English](#english)

</div>

---

## ✨ 特点

- **两档底座，按显卡挑**：4B 版 4bit 量化约 2.5 GB 显存，8B 版约 6 GB，消费级显卡和 Colab 免费 T4 都能跑
- **轻量 adapter**：只发 LoRA 权重，4B 版约 45 MB、8B 版约 61 MB，底座自动从 Hugging Face 拉
- **Colab 零配置体验**：上传 notebook、点运行，十几分钟后拿到一个能直接聊天的网页链接
- **OpenAI 兼容接口**：`tools/serve_v1_1_8b.py` 直接接 AstrBot、NextChat、各种 QQ / 微信机器人框架
- **可选长期记忆**：8B 独立服务可跨会话保存称呼与偏好，支持查看、更新、忘记，按用户独立保存
- **训练全流程公开**：数据清洗、生成、评估脚本和 LLaMA-Factory 配置都在仓库里，可以复现，也可以拿去训你自己的角色
- **思考链实验版**：v1.1-DT 先在内心独白里想，再开口回答

<br>

## 🚀 快速开始

### 方式一：Colab（推荐新手）

notebook 会在 GPU 上加载 **Qwen3-8B + v1.1-R LoRA**（4bit NF4，约 6 GB 显存），再用 Cloudflare 隧道给你一个公网聊天地址。免费 T4 就能跑。

**步骤**

1. 打开 [colab.research.google.com](https://colab.research.google.com) → `File → Upload notebook` → 选 [`colab/ATRI_LoRA_Colab.ipynb`](colab/ATRI_LoRA_Colab.ipynb)
2. `Runtime → Change runtime type` → 选 **T4 GPU**（免费额度够用；A100 / L4 更快）。**换完必须点「重新连接」**，内核才会换成 GPU 版
3. 从上往下依次运行 5 个代码格：
   - 第 1 格：检查 GPU 和显存
   - 第 2 格：装依赖（会卸载 Colab 自带的旧 torchao）
   - 第 3 格：从 Release 下载 adapter 并校验大小、底座
   - 第 4 格：生成前端页面和 FastAPI 后端
   - 第 5 格：加载模型、起服务、开隧道
4. 最后一格会打印 `https://xxx.trycloudflare.com`，浏览器或手机打开就能聊

**换版本**：改第 3 格开头的 `VERSION`

| VERSION | 底座 | 显存（4bit） | 说明 |
|---|---|---|---|
| `v1.1-R` ⭐ | Qwen3-8B | 约 6 GB | 默认，日常聊天 |
| `v1.1-DT` | Qwen3-8B | 约 6 GB | 带思考链，实验版 |
| `v1.0-R` | Qwen3-4B | 约 2.5 GB | 显存不足 6 GB 时用 |
| `v1.0-C` | Qwen3-4B | 约 2.5 GB | 更接近原作说话特征 |

显存不到 6.5 GB 时，notebook 会自动从 v1.1 降到 v1.0-R，不用手动改。

**关于 GH_TOKEN**：仓库已公开，正常不需要配。只有 GitHub API 匿名限流（403）导致下载失败时，才需要在 Colab 左侧 🔑 → Secrets 新增 `GH_TOKEN` 并勾选 Notebook access；不配也能回退成手动上传 zip。

**耗时**：首次运行约 10–20 分钟，大头是下载约 16 GB 的 Qwen3-8B 底座；模型缓存后重新加载约 1–3 分钟。T4 上生成速度约 8–15 token/s。

> [!NOTE]
> 第 2 格卸载 torchao 后如果提示需要重启，`Runtime → Restart session`，然后从第 1 格重新跑。

### 方式二：本地 / 服务器

```bash
git clone https://github.com/NkhszkQAQ/ATRI_LoRA.git
cd ATRI_LoRA
pip install torch transformers peft bitsandbytes accelerate fastapi uvicorn

# 从 Releases 下载 adapter 并解压，得到 adapter/（adapter_model.safetensors + adapter_config.json）
export ATRI_BASE=Qwen/Qwen3-8B
export ATRI_ADAPTER=./adapter
export ATRI_SYSTEM=./persona/system_prompt.txt
python tools/serve_v1_1_8b.py      # 默认监听 127.0.0.1:8003，OpenAI 兼容
```

<br>

## 📦 版本选择

权重放在 [GitHub Releases](https://github.com/NkhszkQAQ/ATRI_LoRA/releases)，按底座模型分组：

| 底座 | 版本 | 一句话 | 资产 |
|---|---|---|---|
| [**Qwen3-8B**](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/Qwen3-8B) | **v1.1-R** ⭐ 推荐 | 招牌反应最扎实，上下文最稳 | `atri_adapter_v1.1-R.zip` |
| | v1.1-DT | 带思考链，实验版（已知问题见 Release 说明） | `atri_adapter_v1.1-DT.zip` |
| [**Qwen3-4B**](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/Qwen3-4B) | v1.0-R | 贴近作品风格，回复完整度最好 | `atri_adapter_v1.0-R.zip` |
| | v1.0-C | 更接近原作说话特征，完整度略逊 | `atri_adapter_v1.0-C.zip` |
| [v0.9-Beta](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/v0.9-Beta) | v0.9-Beta | 原作剧本训练版（旧版，Colab 已在 v1.1 起不再默认使用） | `atri_adapter_v5.zip` |
| [v0.1](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/v0.1) | v0.1 | 早期推理版 | `atri_adapter.zip` |

<sub>后缀：**C** = Classic · **R** = Recommended · **DT** = DeepThinking · **Beta** = 测试版</sub>

**怎么选？** 显存 ≥ 6 GB（含 Colab T4）选 **v1.1-R**；只有 4 GB 左右选 **v1.0-R**；想看她"心里在想什么"试试 **v1.1-DT**。

<br>

## 🔧 规格

| 项 | 值 |
|---|---|
| 底座 | v1.1 起 `Qwen/Qwen3-8B`；v1.0 及以前 `Qwen/Qwen3-4B` |
| 微调方式 | QLoRA，r=32 / alpha=64 / target=`q_proj,v_proj` |
| 产物 | `adapter_model.safetensors`，4B 版约 45 MB / 8B 版约 61 MB |
| 推理精度 | fp16 / 4bit nf4 |
| 显存需求 | 4bit 约 2.5 GB（4B）/ 6 GB（8B） |
| 训练数据 | v0.9-Beta：原作剧本提取 + 人工构造，1,448 条 / 约 42,500 字；v1.0 起另加多轮对话，其用户侧文本来自开发者本人聊天记录的脱敏片段，角色回复由 Claude 生成。**训练数据不随仓库发布** |

<details>
<summary><b>默认推理参数</b>（<code>colab/ATRI_LoRA_Colab.ipynb</code>）</summary>

```yaml
temperature:        0.7
top_p:              0.9
max_new_tokens:     512
repetition_penalty: 1.1
do_sample:          true
seed:               不固定（请求体传 seed 才固定，否则每轮随机）
```

后端为 FastAPI，流式输出（`TextIteratorStreamer`），通过 `cloudflared` 暴露公网地址。

</details>

<br>

## 🧪 自己训练

基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，配置全在 [`training/configs/`](training/configs/)：

```bash
# v1.1（8B）
llamafactory-cli train training/configs/sft_8b_base_x5.yaml

# v0.9（4B）
llamafactory-cli train training/configs/sft_v5_deep.yaml
```

<details>
<summary><b>v1.1 配置对照</b></summary>

| 配置 | 说明 |
|---|---|
| `sft_8b_base_x5.yaml` | 1,448 条原作台词 + 921 条多轮对话，招牌反应样本 ×5 加权，`qwen3_nothink` 模板。v1.1-R 的训练设定与它一致 |
| `sft_8b_c_think*.yaml`<br>`sft_8b_r_think*.yaml`<br>`sft_c_think*.yaml`<br>`sft_r_think*.yaml` | 加训内心独白的实验配置，需要 `qwen3` 模板并开启 `enable_thinking`。v1.1-DT 来自这一类实验，仓库里保留了多份尝试，不保证与发布的权重一一对应 |

</details>

<details>
<summary><b>v0.9 训练流程</b></summary>

1. 从底座训练 2 epoch
2. 从 v4 续训 4 epoch（累计 6）
3. 4bit nf4 量化 + gradient_checkpointing，batch 8 × grad_accum 2

</details>

> [!NOTE]
> 训练数据不随仓库发布。配置里的 `dataset` 名称需要你自己准备同名数据集，并在 `training/data/dataset_info.json` 注册。
> `training/scripts/27_gen_thinking.py` 用于生成内心独白数据，需要自备 OpenAI 兼容接口：设置环境变量 `LLM_API_URL`、`LLM_API_KEY`。

<br>

## 🛰️ 推理服务

**v1.3 服务修复**：继续使用 v1.1-R 权重，增加事实检查、失败重答/来源回退、未知设定处理及多问题的记忆归属修复。已完成 111 项自动化检查、62 题检索回归和真实 8B / 长期记忆组合验收，未发布或部署。部分事实回答依赖来源回退，配置、实测范围与限制见 [v1.3 说明](docs/V1_3.md)。

`tools/serve_v1_1_8b.py` 提供 OpenAI 兼容接口（`/v1/chat/completions`，支持流式），通过环境变量配置：

| 变量 | 作用 | 默认 |
|---|---|---|
| `ATRI_BASE` | 底座模型路径或 HF 名称 | 需自行设置 |
| `ATRI_ADAPTER` | adapter 目录 | 需自行设置 |
| `ATRI_SYSTEM` | system prompt 文件 | 需自行设置 |
| `ATRI_PORT` | 监听端口 | `8003` |
| `ATRI_MODEL_NAME` | 对外暴露的模型名 | `atri-v1.3-8B` |
| `ATRI_SEED` | 已弃用：种子改由请求体 `seed` 控制，不传则随机 | — |

另有 `GET /health`、`GET /v1/models`。三个路径变量的内置默认值是作者服务器上的路径，本地跑必须设置。

可选启用本地剧本 RAG：使用中文 embedding 检索资料，再由现有 Qwen3-8B + LoRA 回答，默认关闭。配置、索引构建与测试见 [RAG 使用说明](docs/RAG.md)。

`tools/serve_v1r.py`（4B）和 `tools/serve_openai.py` 用法类似。DT 版需要 `template: qwen3` 并开启 `enable_thinking`。

<br>

## 🧠 长期记忆

让亚托莉在新会话里也能参考你保存的称呼、偏好与约定。记忆保存在本地数据库中，重启后仍可使用，无需重新训练模型。

**明确保存 · 跨会话保留 · 按用户隔离 · 随时忘记**

开启后，在聊天中单独发送：

```text
/memory remember 称呼=小明
/memory remember 喜欢的颜色=蓝色
```

下次开启新会话再问「我喜欢什么颜色？」，服务会将已保存的记忆交给模型参考。

| 想做什么 | 发送什么 |
|---|---|
| 查看已保存的记忆 | `/memory list` |
| 更新一条记忆 | `/memory remember 喜欢的颜色=绿色` |
| 忘记一条记忆 | `/memory forget 喜欢的颜色` |
| 清空全部记忆 | `/memory clear` |

> [!NOTE]
> 默认关闭，需先在服务器配置并开启。当前仅接入 **8B 独立推理服务**，Colab、4B 和 Gradio 入口不会自动获得此功能。普通聊天不会自动写入记忆；每位用户需要独立密钥，共用密钥就会共用记忆。

**[开始配置长期记忆 →](docs/LONG_TERM_MEMORY.md)** · [客户端怎么填](docs/LONG_TERM_MEMORY.md#3-连接聊天客户端) · [容量与使用范围](docs/LONG_TERM_MEMORY.md#容量与使用范围)

实现随仓库按 Apache-2.0 开源，任何用户均可自行部署；用户记忆与访问密钥不随代码发布。

<br>

## 📁 目录结构

```
training/
  scripts/     提取、清洗、生成、评估脚本
  configs/     LLaMA-Factory 训练配置
  eval/        回归评估题集
persona/       人设（system_prompt.txt + persona_card.md）
tools/         推理服务、命令行 / 网页聊天
colab/         Colab notebook
local/         本地运行（Mac）
webapp/        自建聊天前端
backend/       FastAPI 后端（登录 + 数据库 + 管理后台的完整版）
archive/       按日期归档
docs/          文档
misc/          docker / compose / requirements
```

<br>

## ⚠️ 已知限制

| 项 | 现状 |
|---|---|
| 称呼出现率 | 原作台词 24.3%、训练数据（v4_clean）32.2%，模型输出低于该值。system prompt 已加强制规则，4B 模型遵循度有限 |
| 省略号率 | 原作 42.5%，训练数据 40.5%，模型输出约 10% |
| 场景区分 | 原作日常 / 战斗 / 回忆场景语言风格不同，当前按混合比例训练 |
| 安全拒绝 | 敏感场景拒绝表达弱于 v2.2 |

欢迎在 [Issues](https://github.com/NkhszkQAQ/ATRI_LoRA/issues) 反馈问题、分享对话截图或提改进建议。

<br>

## English

**ATRI LoRA** is a fan-made LoRA adapter that turns Qwen3 (4B / 8B) into a companion chatbot modeled on Atri from the visual novel *ATRI -My Dear Moments-*. The repo ships adapter weights (via Releases), LLaMA-Factory training configs, data-processing scripts, an OpenAI-compatible inference server and a one-click Colab notebook. The model mainly speaks Chinese.

- **Quick try**: upload `colab/ATRI_LoRA_Colab.ipynb` to Colab, pick a T4 GPU, run all cells. Ships with v1.1-R (Qwen3-8B, 4-bit NF4); set `VERSION` in cell 3 to switch to v1.1-DT or a 4B v1.0 variant.
- **Recommended weights**: `v1.1-R` on Qwen3-8B (~6 GB VRAM in 4-bit); `v1.0-R` on Qwen3-4B (~2.5 GB).
- **Serve**: `python tools/serve_v1_1_8b.py` with `ATRI_BASE` / `ATRI_ADAPTER` / `ATRI_SYSTEM` set.
- **Long-term memory**: opt-in, per-user persistent memory for the standalone 8B server. Explicit save/update/forget commands; no automatic collection from ordinary chat. See the [setup guide](docs/LONG_TERM_MEMORY.md).

Unofficial, non-commercial fan project. See [docs/DISCLAIMER.md](docs/DISCLAIMER.md).

<br>

## 📜 声明与许可

个人学习用途，**与《ATRI -My Dear Moments-》版权方无关**，原作素材不得用于商业用途。详见 [docs/DISCLAIMER.md](docs/DISCLAIMER.md)。

代码与配置采用 [Apache License 2.0](LICENSE)。该协议只覆盖本仓库的代码与配置，**不覆盖原作素材**（原作台词、角色立绘、同人表情包等），这些素材的版权归原权利人所有。

<br>

## 🌟 Star History

<a href="https://star-history.com/#NkhszkQAQ/ATRI_LoRA&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=NkhszkQAQ/ATRI_LoRA&type=Date&theme=dark" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=NkhszkQAQ/ATRI_LoRA&type=Date" />
  </picture>
</a>

<div align="center">

如果亚托莉陪你聊得开心，点个 ⭐ 就是对她最好的夸奖——毕竟她是高性能的嘛。

</div>
