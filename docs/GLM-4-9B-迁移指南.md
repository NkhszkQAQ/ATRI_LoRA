# 换 GLM-4-9B 开新一轮训练：操作手册

> 适用：从现在的 Qwen3-4B + adapter v5 迁到 GLM-4-9B-0414 + 全新 LoRA。
> 一句话：**这轮只换基座，数据沿用 v4_clean 不动**——一次只改一个变量，训完才知道提升来自哪。

## 0. 先明确这轮能解决什么

| 问题 | 换 GLM-4-9B 能解决吗 |
|---|---|
| 4B 模型对 prompt 措辞极度敏感，改一句话行为就跳到另一个极端 | 能，9B 的指令跟随稳得多 |
| 输出「像 AI 不像亚托莉」、句长偏长 | 部分能，基座中文语感更强，但主要还看数据 |
| 「螃蟹」等角色口癖少、原台词覆盖率低 | **不能**，这是数据侧的事，留到 v7 补 |
| 省略号率 40.5% 但实际生成只有 10% | 可能改善，9B 更容易学到格式习惯，需要实测 |

所以这一轮的验收标准不是「完美」，而是：**同样 20 道题，GLM 版的称呼率、省略号率、句子长度比 v5 更接近原作，并且不再一改 prompt 就崩。**

## 1. 前置：把机器开起来

1. AutoDL 控制台开机（实例 `atri-gpu`）。
2. 开机后告诉我一声，或者自己跑：
   ```bash
   bash training/scripts/setup_glm_autodl.sh
   ```
   它会查显卡、显存、磁盘、transformers 版本、数据有没有注册，并给出该用哪份配置。

**显存分档**（脚本会自动判断）：
- ≥ 24G（4090 / A5000 / 3090）→ `sft_glm4_9b_qlora.yaml`
- 16G（T4 / 4080 laptop）→ `sft_glm4_9b_qlora_16g.yaml`
- < 16G → 9B QLoRA 跑不动，这轮就先别换

磁盘至少留 40G：模型 19G + checkpoint 约 1G + 数据缓存。

## 2. 下载基座（约 19G）

```bash
bash training/scripts/setup_glm_autodl.sh --download
# 等价于：
modelscope download --model ZhipuAI/GLM-4-9B-0414 --local_dir /root/autodl-tmp/models/GLM-4-9B-0414
```

为什么不继续用 `glm-4-9b-chat`（2024 版）：0414 是 2025 年的新版权重，中文指令跟随更好，而且用 transformers 内置实现（≥ 4.51），不依赖 `trust_remote_code` 加载远端代码。老版仍然可用，把 yaml 里 `model_name_or_path` 换成 `zai-org/glm-4-9b-chat` 即可。

## 3. 数据：这轮不动

沿用 `atri_v4_clean`（1448 条，平均 15.7 字、省略号 40.5%、称呼 30.6%）。仓库里的 `dataset_info.json` 之前漏注册了这个数据集，这次已经补上，把它同步到服务器的 `/root/autodl-tmp/atri/data/dataset_info.json`。

**不要**同时改数据。换基座已经是最大的变量，再改数据就分不清提升来自哪边。口癖补齐（原作 12 条螃蟹台词只剩 8 条这类）留到 v7。

## 4. 开训

```bash
cd /root/autodl-tmp/LlamaFactory        # 你服务器上的 llamafactory 目录
llamafactory-cli train /root/autodl-tmp/atri/configs/sft_glm4_9b_qlora.yaml
```

仓库里有两份现成配置，路径按服务器实际情况改：
- `training/configs/sft_glm4_9b_qlora.yaml`（24G，batch 4×4，lr 8e-5，3 epoch）
- `training/configs/sft_glm4_9b_qlora_16g.yaml`（16G，batch 1×16，cutoff 1536）

预期：约 250 步、1.5~2 小时（24G 卡）。输出到 `/root/autodl-tmp/atri-runs/sft-v6-glm4-9b`。

## 5. 三个必踩的坑（照抄 Qwen 配置会出事）

1. **`template` 必须是 `glm4`**
   GLM 的对话格式是 `[gMASK]<sop><|system|>…<|user|>…<|assistant|>`，跟 Qwen 的 `<|im_start|>` 完全两套。写成 `qwen3_nothink` 会让模型学到不存在的特殊 token，训练 loss 正常但推理全是乱码。

2. **`lora_target` 用 `all`**
   GLM 的注意力层叫 `self_attention.query_key_value` / `self_attention.dense`，不是 `q_proj` / `v_proj`。照抄 `q_proj,v_proj` 会匹配到 0 个模块——要么直接报错，要么静默训了个空 adapter（训完 adapter_model.safetensors 只有几百 KB，就是这个症状）。想精确控制就写 `query_key_value,dense`。

3. **旧的 Qwen adapter 一律不能用**
   基座换了，隐层和词表都不同（Qwen 151936 vs GLM 151552），必须从基座从零训 LoRA。也不要设 `adapter_name_or_path`。

另外两点：
- GLM-4-9B 没有 thinking 模式，别加 `enable_thinking` 参数。
- 训练和推理必须用**同一个** `glm4` 模板。推理端（Colab / `local/run_local.py`）改成 `template=glm4`，或者用 `llamafactory-cli chat --template glm4`，两边不一致会出现「训练时学到的格式，推理时喂不进去」。

## 6. 验收：跑同一套题，和 v5 比

固定测试集在 `training/eval/atri_bench.jsonl`，20 道题覆盖称呼、被贬低、螃蟹、安慰、身份、安全边界。

```bash
# 先跑现役 v5 做基线（本机就能跑）
python training/scripts/21_bench.py --base Qwen/Qwen3-4B --adapter models/adapter_v22 --tag qwen-v5

# 训练完跑 GLM 版（服务器上跑，或把 adapter 拉回本机）
python training/scripts/21_bench.py --base /root/autodl-tmp/models/GLM-4-9B-0414 \
    --adapter /root/autodl-tmp/atri-runs/sft-v6-glm4-9b --tag glm-v6
```

看四个数：平均字数（原作 13.2，越低越像）、省略号率、称呼率、复读嫌疑数（≤6 字的回复条数）。结果存 `training/eval/results/*.json`。

**人工再补一眼**：同一道题两个版本的输出并排放，判断「这段像不像亚托莉」——指标只能排除明显退化的版本，最终还得眼睛看。

## 7. 训完之后

- adapter 大约 100~200MB（`lora_target: all` 比只训 q/v 大一些）。
- Colab T4 16G 跑 9B 4bit 推理没问题（权重约 6G + KV cache），把 `ATRI_LoRA_Colab.ipynb` 里的 base 换成 GLM、template 换成 `glm4` 即可。
- 本机 RTX 4070 8G 跑 9B 4bit 推理勉强（约 6.5G，显存吃满），建议还是走 Colab。

## 8. 时间线

| 步骤 | 耗时 |
|---|---|
| 开机 + 检查 | 5 分钟 |
| 下载模型 | 10~20 分钟（看网速） |
| 数据同步 + 开训 | 1.5~2 小时 |
| 跑固定测试集 | 10 分钟 |
| 部署到 Colab 复测 | 20 分钟 |

一轮下来半天内能出结论。
