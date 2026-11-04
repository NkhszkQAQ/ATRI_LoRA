#!/usr/bin/env bash
# GLM-4-9B-0414 训练前置检查 + 模型下载（在 AutoDL 服务器上跑，不删任何东西）
# 用法：
#   bash training/scripts/setup_glm_autodl.sh          # 只检查
#   bash training/scripts/setup_glm_autodl.sh --download   # 检查并下载模型
set -uo pipefail

MODEL_DIR=${MODEL_DIR:-/root/autodl-tmp/models/GLM-4-9B-0414}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/atri/data}
fail=0
ok()   { printf '  [OK]   %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=1; }

echo "== 1. GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
  vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${vram:-0}" -ge 23000 ]; then
    ok "显存 ${vram}MB → 用 sft_glm4_9b_qlora.yaml（24G 版）"
  elif [ "${vram:-0}" -ge 15000 ]; then
    warn "显存 ${vram}MB → 9B 4bit 偏紧，用 sft_glm4_9b_qlora_16g.yaml"
  else
    bad "显存 ${vram}MB → 9B QLoRA 跑不动，先换机器或降到 4B"
  fi
else
  bad "没有 nvidia-smi，不在 GPU 机器上"
fi

echo "== 2. 磁盘 =="
df -h /root/autodl-tmp 2>/dev/null | tail -1 || df -h /root | tail -1
avail=$(df -BG --output=avail /root/autodl-tmp 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${avail:-}" ] && [ "$avail" -lt 40 ]; then
  bad "数据盘剩余 ${avail}G，模型+权重+checkpoint 至少留 40G"
else
  ok "数据盘空间够（模型约 19G，checkpoint 约 1G）"
fi

echo "== 3. Python 环境 =="
python - <<'PY'
import importlib, sys
print(f"  python {sys.version.split()[0]}")
for name, need in [("torch", "2.1"), ("transformers", "4.51"), ("peft", "0.10"), ("bitsandbytes", "0.43"), ("llamafactory", "0.9")]:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print(f"  [OK]   {name} {v}")
    except Exception as e:
        print(f"  [FAIL] {name} 未安装：{e}")
PY
python -c "import transformers; from packaging.version import parse; assert parse(transformers.__version__) >= parse('4.51.0')" 2>/dev/null \
  && ok "transformers ≥ 4.51，支持 GLM-4-9B-0414 内置结构" \
  || bad "transformers 低于 4.51：pip install -U 'transformers>=4.51,<5'"

echo "== 4. 训练数据 =="
for f in "$DATA_DIR/v2/pilot/atri_v4_clean.jsonl" "$DATA_DIR/dataset_info.json"; do
  [ -f "$f" ] && ok "$f ($(wc -l < "$f" 2>/dev/null || echo '?') 行)" || bad "缺文件：$f"
done
grep -q 'atri_v4_clean' "$DATA_DIR/dataset_info.json" 2>/dev/null \
  && ok "dataset_info.json 已注册 atri_v4_clean" \
  || bad "dataset_info.json 未注册 atri_v4_clean（把仓库里那份同步过去）"

echo "== 5. 基座模型 =="
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
  du -sh "$MODEL_DIR" | awk '{print "  "$1"  "$0}' | head -1
  ok "已存在：$MODEL_DIR"
else
  warn "未下载：$MODEL_DIR"
  if [ "${1:-}" = "--download" ]; then
    echo "  开始下载（约 19G，国内走 ModelScope 更快）..."
    (modelscope download --model ZhipuAI/GLM-4-9B-0414 --local_dir "$MODEL_DIR" \
      || HF_ENDPOINT=https://hf-mirror.com huggingface-cli download zai-org/GLM-4-9B-0414 --local-dir "$MODEL_DIR" \
      || echo "  [FAIL] 两个源都失败，手动指定镜像后重试")
  else
    echo "  → 加 --download 参数，或手动："
    echo "     modelscope download --model ZhipuAI/GLM-4-9B-0414 --local_dir $MODEL_DIR"
  fi
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "== 检查通过，下一步 =="
  echo "  llamafactory-cli train training/configs/sft_glm4_9b_qlora.yaml"
  echo "  （16G 卡换 sft_glm4_9b_qlora_16g.yaml）"
else
  echo "== 有 FAIL 项，先按上面提示修"
fi
exit "$fail"
