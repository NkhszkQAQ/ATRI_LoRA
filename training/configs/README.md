# 底座模型与硬件档位

**显存修正：本项目将 RTX 4070 / 5060 均按 ~8GB 档处理。**  
默认底座：**`Qwen3-4B`（Ollama：`qwen3:4b`）**。  
`Qwen3-8B` 仅在探测到显存 **>10GB** 或手动选择时使用；14B 高端少用。

均使用官方 Instruct/Chat 向权重（HF：`Qwen/Qwen3-*`），Apache-2.0。角色陪伴场景默认 **关闭 thinking**（配置里 `template: qwen3_nothink`）。

## 档位矩阵

| 档位 | 显存参考 | 训练 / 在线底座 | Ollama 标签 | 配置 |
|------|----------|-----------------|-------------|------|
| **默认（4070 / 5060）** | ~8GB | **Qwen3-4B** LoRA | `qwen3:4b` | `sft_qwen3_4b_lora.yaml` |
| 可选 | >10GB 或 QLoRA | Qwen3-8B | `qwen3:8b` | `sft_qwen3_8b_lora.yaml` / `*_qlora.yaml` |
| 高端（少用） | ≥48GB | Qwen3-14B LoRA | — | `sft_qwen3_14b_lora.yaml` |

> 不要在 8–12GB 卡上默认拉 32B。32B 已移出推荐路径。

## 推荐命令

```bash
# 5060：生成 + 训练 4B
# vllm serve Qwen/Qwen3-4B --port 8000
python scripts/02_gen_synthetic.py --model Qwen/Qwen3-4B --workers 4
llamafactory-cli train configs/sft_qwen3_4b_lora.yaml

# 4070：默认路径（8B）
# vllm serve Qwen/Qwen3-8B --port 8000
python scripts/02_gen_synthetic.py --model Qwen/Qwen3-8B --workers 8
llamafactory-cli train configs/sft_qwen3_8b_lora.yaml

# 高端（少用）
llamafactory-cli train configs/sft_qwen3_14b_lora.yaml
```

Ollama（Docker）时把 `--base-url` 换成 `http://localhost:11434/v1`，`--model` 换成 `ollama list` 中的名字（如 `qwen3:4b` / `qwen3:8b`）。

## 权重下载（按需，勿提交进 git）

```bash
# Hugging Face（示例）
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
# 高端可选
# huggingface-cli download Qwen/Qwen3-14B --local-dir models/Qwen3-14B
```

本地目录可写进 yaml 的 `model_name_or_path: models/Qwen3-8B`。`models/` 应自行加入 ignore（大文件不进仓库）。
