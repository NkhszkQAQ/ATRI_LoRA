# 本地运行 ATRI（Mac）

## 需要什么

| 项 | 要求 |
|---|---|
| Python | 3.11 – 3.14 |
| 内存 | 16 GB 起（4B 模型 fp16 约占 8 GB） |
| 磁盘 | 基座模型 8 GB + adapter 45 MB |
| 显卡 | 不需要，Apple Silicon 用 MPS 加速即可 |

## 文件放哪

```
atri-lora/
├── 我的东西/
│   ├── 电脑上跑/                  ← run_local.py、这个说明、本地 notebook
│   └── 模型/
│       ├── adapter_v22/           ← v2.2 adapter（新，默认使用）
│       └── adapter_旧版v2pilot/   ← 旧版 adapter
└── persona/system_prompt.txt      ← 人格设定
```

## 第一次运行

```bash
cd ~/atri-lora/我的东西/电脑上跑

# 建虚拟环境（用 uv，比 pip 快）
uv venv .venv-torch --python 3.12
source .venv-torch/bin/activate

# 装依赖（torch 约 2 GB，第一次要几分钟）
uv pip install torch transformers peft accelerate
```

> 如果 Python 3.14 装 torch 报错，就用 `--python 3.12` 建环境。

## 启动

```bash
cd ~/atri-lora/我的东西/电脑上跑
python run_local.py                                        # 用 v2.2
python run_local.py "../模型/adapter_旧版v2pilot"           # 用旧版
```

第一次会从 HuggingFace 下载 Qwen3-4B（8 GB），之后走缓存。

## 速度预期（M5 16 GB）

| 情况 | 大概速度 |
|---|---|
| Apple Silicon MPS | 8–15 字/秒 |
| 纯 CPU | 1–3 字/秒 |

一句 30 字的回复大约 2–4 秒（MPS）。想更快就得回 Colab 用 T4/A100。

## 关于「云端有训练数据」

云端下载的是 **adapter 权重**，不是训练数据。

- 训练数据（`data/v2/*.jsonl`）只在**训练**时用
- 训练完，效果已经压进 `adapter_model.safetensors` 这 45 MB 里
- 推理时只需要：基座 + adapter + system prompt，**不需要任何数据文件**

所以你下载 v2.2 的原因是「它是更新的训练结果」，不是因为「它带了数据」。

## 两套 adapter 的差别

| | v2 pilot | v2.2 |
|---|---|---|
| 位置 | `adapter/` | `adapter_v22/` |
| 时间 | 10-26 | 10-31 |
| 结构 | Qwen3-4B / r=32 / alpha=64 / q_proj,v_proj | 完全相同 |
| 差别 | 训练数据/轮次不同，权重不同（sha256 不同） | |

切换只需要改 `run_local.py` 的参数或改 `ADAPTER` 常量。

## 常见问题

| 现象 | 处理 |
|---|---|
| `MPS backend out of memory` | 关掉其他吃内存的 App；或改用 CPU（慢） |
| 下载模型很慢 | 设 `export HF_ENDPOINT=https://hf-mirror.com` |
| 回复被截断 | `run_local.py` 里 `max_new_tokens` 调大（256 → 512） |
| 想换称呼 | 改 `run_local.py` 里 `SYSTEM` 的替换部分 |

## 和 Colab 的关系

同一套权重、同一套 system prompt，只是设备不同：

```
Colab:  T4/A100 → fp16 → 快（20–40 字/秒）
本地:   M5 MPS  → fp16 → 中（8–15 字/秒）
```

本地的好处是不用等 Colab 启动、不用云盘、不怕断线；坏处是速度慢一截。
