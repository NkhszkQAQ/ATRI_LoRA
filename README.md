# ATRI LoRA

基于《ATRI -My Dear Moments-》角色亚托莉的 Qwen3-4B 陪伴聊天 LoRA——权重、训练脚本与 Colab 推理 notebook。

## 规格

| 项 | 值 |
|---|---|
| 基座 | v1.1 起 `Qwen/Qwen3-8B`；v1.0 及以前 `Qwen/Qwen3-4B` |
| 微调 | QLoRA，LoRA r=32 / alpha=64 / target=q_proj,v_proj |
| 产物 | `adapter_model.safetensors`，4B 版约 45 MB / 8B 版约 61 MB |
| 训练数据 | v0.9-Beta：原作剧本提取 + 人工构造，1,448 条 / 约 42,500 字；v1.0 起另加多轮对话，其用户侧文本来自开发者本人聊天记录的脱敏片段，角色回复由 Claude 生成。训练数据不随仓库发布 |
| 推理精度 | fp16 / 4bit nf4 |
| 显存需求 | 4bit 约 2.5 GB（4B）/ 6 GB（8B） |

## 使用

### 1. 克隆仓库

```bash
git clone https://github.com/NkhszkQAQ/ATRI_LoRA.git
cd ATRI_LoRA
```

### 2. 下载权重

权重放在 GitHub Releases，不进 git：

| 版本 | 说明 | 资产 |
|---|---|---|
| [v1.1-8B](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/v1.1-8B)（推荐） | 8B 底座，招牌反应最扎实，上下文最稳 | `atri_adapter_v1.1-8B.zip` |
| [v1.1-DT](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/v1.1-DT) | 带思考链，实验版（已知问题见说明） | `atri_adapter_v1.1-DT.zip` |
| [v1.0-R](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/v1.0-R) | 4B 底座，贴近作品风格，回复完整度最好 | `atri_adapter_v1.0-R.zip` |
| [v1.0-C](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/v1.0-C) | 4B 底座，更接近原作说话特征，完整度略逊 | `atri_adapter_v1.0-C.zip` |
| [v0.9-Beta](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/v0.9-Beta) | 原作剧本训练版，Colab notebook 默认加载 | `atri_adapter_v5.zip` |
| [v0.1](https://github.com/NkhszkQAQ/ATRI_LoRA/releases/tag/v0.1) | 早期推理版 | `atri_adapter.zip` |

均为约 42–44 MB 的 zip，解压后得到 `adapter/`（`adapter_model.safetensors` + `adapter_config.json`）。

### 3. 在 Colab 运行

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. `File → Upload notebook`，选择 `colab/ATRI_Colab_v5.ipynb`
3. `Runtime → Change runtime type`，选择 T4 GPU
4. 依次运行代码单元格；第 3 格会先用 Colab Secrets 里的 `GH_TOKEN` 从 Release 自动下载 adapter（`v0.9-Beta`），没配 token 就回退成手动上传 zip
5. 最后一个单元格输出 `https://xxx.trycloudflare.com`，浏览器打开

想走自动下载：本地跑 `gh auth token` 拿到 `gho_...`，Colab 左侧 🔑 → Secrets 新增 `GH_TOKEN`，勾选 Notebook access。不配也能跑，只是要手动传 zip。

首次运行约 5-8 分钟，主要耗时在下载基座模型。

## 推理参数

`colab/ATRI_Colab_v5.ipynb` 中的默认值：

```yaml
temperature:        0.85
top_p:              0.9
max_new_tokens:     512
repetition_penalty: 1.05
do_sample:          true
```

后端为 FastAPI，流式输出（`TextIteratorStreamer`），通过 `cloudflared` 暴露公网地址。

## 训练

配置文件在 `training/configs/`，使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：

```bash
llamafactory-cli train training/configs/sft_v5_deep.yaml
```

v5 训练流程：

1. 从基座训练 2 epoch
2. 从 v4 续训 4 epoch（累计 6）
3. 4bit nf4 量化 + gradient_checkpointing，batch 8 × grad_accum 2

后续 v1.0 版本在脱敏语料上继续训，具体见对应 Release 说明。

## 目录结构

```
training/          训练
  data/            数据集（jsonl）
  scripts/         提取、清洗、生成、评估脚本
  configs/         LLaMA-Factory 训练配置
  eval/            回归评估题集
persona/           人设（system_prompt.txt + persona_card.md）
webapp/            自建聊天前端
backend/           FastAPI 后端服务
colab/             Colab notebook
local/             本地运行（Mac）
models/            训练好的 adapter
archive/           按日期归档
docs/              文档
misc/              其他（docker / compose / requirements）
```

## 已知限制

| 项 | 数据 |
|---|---|
| 称呼出现率 | 原作台词 24.3%、训练数据（v4_clean）32.2%，model 输出低于该值。system prompt 已加强制规则，4B 模型遵循度有限 |
| 省略号率 | 原作 42.5%，训练数据 40.5%，模型输出约 10% |
| 场景区分 | 原作日常/战斗/回忆场景语言风格不同，当前按混合比例训练 |
| 安全拒绝 | 敏感场景拒绝表达弱于 v2.2 |

## 声明

个人学习用途。与《ATRI -My Dear Moments-》版权方无关，原作素材不得用于商业用途。详见 [docs/DISCLAIMER.md](docs/DISCLAIMER.md)。

## License

代码与配置部分采用 Apache License 2.0，见 [LICENSE](LICENSE)。
该协议只覆盖本仓库的代码与配置，**不覆盖原作素材**（原作台词、角色立绘、同人表情包等），这些素材的版权归原权利人所有。
