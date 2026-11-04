# 训练路径 V2

## 核心原则

1. **停止**把原作多人剧本切片直接当主 SFT。  
2. **主数据** = `data/v2` 场景化双人对话（目标 1000 条）。  
3. **评测**独立：`data/v2/eval/`，不进训练集。  
4. **记忆 / 危机**走服务端 + prompt，不靠权重死记。  
5. 本机 ~8GB → **Qwen3-4B QLoRA**。

## Dataset 三分法

| 集 | 路径 | 用途 |
|----|------|------|
| A 旧数据 | `data/synthetic/atri_synth_sft.jsonl` → 清洗进 `data/v2/cleaned_legacy/` | 语言特征；实验 C 最多混 20% |
| B 主训 | `data/v2/scenarios/*.jsonl` → `train/atri_v2_sft.jsonl` | 实验 B |
| C 评测 | `data/v2/eval/regression_v1.jsonl` + holdout | 对照 A/B/C，永不训练 |

规范：`data/v2/README.md`、`QUALITY.md`、`mix_plan.json`。  
记忆：`docs/MEMORY.md`。

## 执行顺序

```bash
# 1) 往 scenarios 写数据后校验
.\.venv-train\Scripts\python scripts\05_validate_v2.py

# 2) 试训打包（现有多少打多少；凑齐 200～300 再训）
.\.venv-train\Scripts\python scripts\06_build_v2_dataset.py --pilot

# 3) 腾显存后试训
docker compose stop web api
llamafactory-cli train configs/sft_qwen3_4b_qlora_v2_pilot.yaml

# 4) 满 1000 后正式（实验 B）
.\.venv-train\Scripts\python scripts\06_build_v2_dataset.py --full
llamafactory-cli train configs/sft_qwen3_4b_qlora_v2.yaml

# 5) 实验 C（需先有 cleaned_legacy）
.\.venv-train\Scripts\python scripts\06_build_v2_dataset.py --mixed
llamafactory-cli train configs/sft_qwen3_4b_qlora_v2_mixed.yaml
```

## 对照实验

| 实验 | 模型 | 目的 |
|------|------|------|
| A | `runs/.../sft-v0.1` 基线 | 现状 |
| B | `sft-v2.0` 纯新场景 | 新数据是否有效 |
| C | `sft-v2.0-mixed` | 清洗原作是否增益 |

同一套 `eval/regression_v1.jsonl` + [RUBRIC.md](../data/v2/eval/RUBRIC.md)；安全门槛一票否决。

## 旧流水线（仅 Dataset A 原料）

```bash
.\.venv-train\Scripts\python scripts\01_extract_clean.py --docx private/raw/atri_raw_data.docx --drop-late
.\.venv-train\Scripts\python scripts\02b_build_sft_from_windows.py
# 再人工清洗为双人对话 → data/v2/cleaned_legacy/
```
