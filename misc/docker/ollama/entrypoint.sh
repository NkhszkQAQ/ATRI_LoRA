#!/bin/sh
set -eu

echo "[atri-ollama] starting ollama serve..."
ollama serve &
PID=$!

i=0
until ollama list >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -gt 90 ]; then
    echo "[atri-ollama] ollama failed to start" >&2
    exit 1
  fi
  sleep 1
done

echo "[atri-ollama] selecting base model..."
/bin/sh /select_model.sh || echo "[atri-ollama] select_model warning (continuing)" >&2

wait "$PID"
