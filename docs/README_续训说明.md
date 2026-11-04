# ATRI V2 Pilot LoRA — 交接续训说明

## 包内有什么

```text
adapter/          # 已训完的 QLoRA adapter（约 45MB）
  adapter_model.safetensors
  adapter_config.json
data/
  dataset_info.json
  v2/             # 场景数据 + pilot 打包集 + 质检文档
configs/          # 本机用过的 yaml（参考）
sft_continue_from_pilot.yaml   # 推荐：在本权重上开新一轮
```

**不含**基座模型。对方机器需自行准备 `Qwen/Qwen3-4B`（HuggingFace 或本地路径）。

## 当前权重关键信息

| 项 | 值 |
|----|-----|
| 基座 | `Qwen/Qwen3-4B` |
| 方法 | QLoRA 4bit + LoRA |
| LoRA | `r=32`, `alpha=64`, `target=q_proj,v_proj` |
| 模板 | `qwen3_nothink` |
| 已完成 | pilot 2 epoch / 98 steps，`train_loss≈1.81` |
| 数据注意 | pilot 里仍有字面 `{{user}}` / 旧 system，正式专训前建议先清洗再训 |

## 环境（对方服务器）

```bash
# Python 3.11+，CUDA 可用
pip install "llamafactory[torch,metrics]"  # 或按你们现有 venv 安装
# 需要: torch + peft + transformers + bitsandbytes（Linux 更稳）
```

确认能 `import torch; torch.cuda.is_available()`。

## 目录怎么放

假设解压到 `/work/atri-handoff`：

```text
/work/atri-handoff/
  adapter/
  data/dataset_info.json
  data/v2/...
  sft_continue_from_pilot.yaml
```

把 `data/` 整目录当作 LLaMA-Factory 的 `dataset_dir`（即其下直接有 `dataset_info.json` 和 `v2/`）。

## 开训（推荐：在 adapter 上继续新一轮）

```bash
cd /work/atri-handoff

# 1) 如需改数据：先改 data/v2/scenarios，再在原仓库跑 06_build 或直接替换 pilot jsonl
# 2) 编辑 sft_continue_from_pilot.yaml：改 model 路径、output_dir、epoch、显存相关
# 3) 启动
llamafactory-cli train sft_continue_from_pilot.yaml
```

关键配置项（已写在 `sft_continue_from_pilot.yaml`）：

```yaml
model_name_or_path: Qwen/Qwen3-4B          # 或本地基座路径
adapter_name_or_path: ./adapter            # 从本包权重接着训
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_target: q_proj,v_proj                 # 必须与 adapter_config 一致
quantization_bit: 4
dataset_dir: ./data
dataset: atri_v2_pilot                     # 或你们注册的新数据集名
template: qwen3_nothink
```

**不要**对已训完的 final adapter 使用 `resume_from_checkpoint`（那是同一趟中断续跑用的）。  
要用「接着训」就用 `adapter_name_or_path`。

## 显存建议

| 显存 | 建议 |
|------|------|
| 8GB | `cutoff_len: 512`，`batch=1`，`grad_accum=8~16`，关 eval，LoRA 保持 q/v |
| 16GB+ | 可把 `cutoff_len` 提到 1024/1536，或 `lora_target: all`（那是新结构，不能直接加载本 adapter） |
| 24GB+ | 可考虑正式配比 `sft_qwen3_4b_qlora_v2.yaml` 风格拉长上下文 |

Windows 本机曾因 Wallpaper Engine + 近满显存蓝屏；服务器上尽量独占 GPU。

## 训完产物

输出目录里会有新的 `adapter_model.safetensors`。推理时：

```text
基座 Qwen3-4B + 新 adapter 目录
```

可用原项目 `scripts/serve_lora.py`（需改 `--lora` / adapter 路径）挂服务。

## 数据清洗提醒（强烈建议下一轮先做）

1. GPT 回复里的 `{{user}}` → 换成具体名（或统一假名再线上替换）  
2. system 换成现行 `data/v2/SYSTEM_TEMPLATE.txt`，并填好 `{{memory}}`  
3. 去掉过短样本，补 `multi_turn`  

清洗后再 `dataset_info.json` 注册新文件名开训。

## 校验清单

- [ ] CUDA 正常  
- [ ] 已下载/挂载 `Qwen/Qwen3-4B`  
- [ ] `adapter_config.json` 的 `r/alpha/target_modules` 与 yaml 一致  
- [ ] `data/dataset_info.json` 里的 `file_name` 相对 `dataset_dir` 能找到  
- [ ] `output_dir` 写在有足够磁盘的路径  
