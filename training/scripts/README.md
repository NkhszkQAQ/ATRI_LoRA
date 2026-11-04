# Scripts

按顺序运行。依赖见根目录 `requirements.txt`（数据脚本至少需要 `python-docx`、`openai`）。  
底座档位见 [`configs/README.md`](../configs/README.md)（**4070/5060 ~8GB → Qwen3-4B**）。  
在线对话请用 Docker 栈（见 [`docs/OPS.md`](../docs/OPS.md)）。

| 脚本 | 职责 | 何时跑 |
|------|------|--------|
| `01_extract_clean.py` | 从 `private/raw/*.docx` 提取 → `private/derived/` | Dataset A 原料 |
| `02_gen_synthetic.py` | 本地 OpenAI 兼容服务生成 → `data/synthetic/raw.jsonl` | 可选 |
| `02b_build_sft_from_windows.py` | 窗口 → 旧版 sharegpt | 仅遗留 |
| `03_filter_dedup.py` | 过滤去重泄漏检查 + safety_seed | 接在 02 后 |
| `04_build_dpo.py` | 构造 DPO 对 | 做 DPO 时 |
| `05_validate_v2.py` | 校验 `data/v2/scenarios` schema/质检 | 写数据后必跑 |
| `06_build_v2_dataset.py` | 打包 pilot / 正式 / mixed | 训练前 |
| `07_v2_entry.py` | 交互式录入一条对话 | 手写补数据 |
| `07b_gen_v2_batch.py` | 模板批量扩写（质量一般，已不推荐作主数据） | 可选 |
| `07c_sim_hq_dialogues.py` | **按剧本口吻库模拟高质量双人对话并重写 scenarios** | 主推 |
| `serve_lora.py` | 本机挂 LoRA 推理 :8008 | 线上对话 |

```bash
# === V2 推荐 ===
python scripts/05_validate_v2.py
python scripts/06_build_v2_dataset.py --pilot
llamafactory-cli train configs/sft_qwen3_4b_qlora_v2_pilot.yaml

# 数据凑齐后
python scripts/06_build_v2_dataset.py --full
llamafactory-cli train configs/sft_qwen3_4b_qlora_v2.yaml
```

详见 [`docs/TRAIN.md`](../docs/TRAIN.md)、[`data/v2/README.md`](../data/v2/README.md)。
