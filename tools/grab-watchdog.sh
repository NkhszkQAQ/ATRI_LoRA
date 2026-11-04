#!/bin/bash
# 抢卡看门狗：抢卡进程意外退出时自动拉起；已抢到（开机成功）则退出。
# 用法：nohup bash tools/grab-watchdog.sh <日志绝对路径> <实例ID> >/dev/null 2>&1 &
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ" || exit 1
LOG="$1"
INSTANCE="$2"
if [ -z "$LOG" ] || [ ! -f "$LOG" ] || [ -z "$INSTANCE" ]; then
  echo "watchdog: 需要传入存在的日志路径和实例ID" >&2
  exit 1
fi

MAX_RESTART=8
restarts=0

while true; do
  sleep 60
  # 还活着就继续盯
  if pgrep -f 'autodl-grab.py grab' >/dev/null 2>&1; then
    continue
  fi
  # 已抢到（开机成功）→ 收工，不再重启
  if grep -q "✅ 开机成功" "$LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] 已抢到卡，watchdog 退出" >> "$LOG"
    exit 0
  fi
  if [ "$restarts" -ge "$MAX_RESTART" ]; then
    echo "[$(date '+%F %T')] watchdog：已重启 $MAX_RESTART 次仍无进程，放弃" >> "$LOG"
    exit 1
  fi
  restarts=$((restarts + 1))
  echo "[$(date '+%F %T')] watchdog：抢卡进程不见了，第 $restarts 次重启" >> "$LOG"
  nohup python3 -u tools/autodl-grab.py grab "$INSTANCE" --yes >> "$LOG" 2>&1 &
  NEWPID=$!
  nohup caffeinate -imsu -w "$NEWPID" >/dev/null 2>&1 &
done
