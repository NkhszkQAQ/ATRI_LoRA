#!/bin/sh
# Detect VRAM, propose Qwen3 tag, reuse or pull. 4070/5060 (~8GB) → qwen3:4b
set -eu

PREFERRED_4B="qwen3:4b"
PREFERRED_8B="qwen3:8b"

vram_mb=""
if command -v nvidia-smi >/dev/null 2>&1; then
  vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
fi
echo "[atri-model] detected VRAM_MB=${vram_mb:-unknown}"

if [ -n "${MODEL_FORCE:-}" ] && [ "${MODEL_FORCE}" != "none" ]; then
  choice="$MODEL_FORCE"
elif [ -n "${vram_mb}" ] && [ "$vram_mb" -gt 10240 ] 2>/dev/null; then
  choice="$PREFERRED_8B"
else
  choice="${ATRI_MODEL:-$PREFERRED_4B}"
fi

seconds="${MODEL_CONFIRM_SECONDS:-10}"
# Docker usually has no TTY on entrypoint → auto-accept
if [ ! -t 0 ]; then
  seconds=0
fi

echo "============================================================"
echo "[atri-model] proposed base model: $choice"
echo "[atri-model] alternatives: $PREFERRED_4B , $PREFERRED_8B"
echo "[atri-model] waiting ${seconds}s (Enter accepts / type tag to override)"
echo "============================================================"

if [ "$seconds" -gt 0 ]; then
  if read -r -t "$seconds" typed 2>/dev/null; then
    if [ -n "${typed:-}" ]; then
      choice="$typed"
    fi
  fi
fi

echo "[atri-model] selected: $choice"

if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$choice" \
  || ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -q "^${choice}"; then
  echo "[atri-model] reuse local model: $choice"
else
  echo "[atri-model] pulling $choice ..."
  ollama pull "$choice"
fi

echo "$choice" >/tmp/atri_active_model
echo "[atri-model] ACTIVE_MODEL=$choice"
