#!/usr/bin/env bash
# 一键起服务：设备侧守护链（任一段挂掉自动整链重启）+ adb 端口转发 + PC 侧 HTTP 前端
#
#   ./start_server.sh            # 起服务并占用前台，Ctrl-C 只停 HTTP 前端
#   ./start_server.sh 9000       # 换 HTTP 端口
#
# 设备侧手动停：adb shell "touch /data/local/tmp/sup.stop; pkill -f llama-split"
set -e

ADB="${ADB:-adb}"
HTTP_PORT="${1:-7788}"
DEV_PORT=18080
NEED_PROCS=4          # parent + 3 stages
# v2 pipeline (serve9b.sh defaults): --slots 2 --batch 64. Override from the environment.
export CTX="${CTX:-8192}" BATCH="${BATCH:-64}" SLOTS="${SLOTS:-2}" EXTRA="${EXTRA:-}"

# adb is not always on PATH (on Windows it usually sits next to platform-tools), so fail with a
# usable message instead of bash's bare "line N: adb: command not found".
command -v "$ADB" > /dev/null 2>&1 || {
  echo "adb not found: '$ADB'" >&2
  echo "  set ADB=/path/to/adb, or put platform-tools on your PATH" >&2
  exit 127
}
"$ADB" get-state > /dev/null

count_procs() {
  "$ADB" shell "ps -ef | grep -c '[l]lama-split'" 2>/dev/null | tr -d '\r' | head -1
}

n=$(count_procs || echo 0)
if [ "${n:-0}" -ge "$NEED_PROCS" ]; then
  echo "device: chain already up ($n procs)"
else
  echo "device: (re)starting 3-stage chain (slots=$SLOTS batch=$BATCH ctx=$CTX), ~10 s to load"
  "$ADB" shell "touch /data/local/tmp/sup.stop; for p in \$(ps -ef | grep '[l]lama-split' | awk '{print \$2}'); do kill -9 \$p; done" >/dev/null 2>&1 || true
  sleep 1
  "$ADB" shell "rm -f /data/local/tmp/sup.stop; setsid nohup sh /data/local/tmp/sup9b.sh $DEV_PORT > /dev/null 2>&1 < /dev/null &" >/dev/null
  for _ in $(seq 1 40); do
    if [ "$(count_procs)" -ge "$NEED_PROCS" ]; then break; fi
    sleep 1
  done
  echo "device: chain up ($(count_procs) procs)"
fi

# Windows lets two processes listen on the same port (SO_REUSEADDR); the stale one then answers
# with old code and you debug a fix that is already in place. Clear the port first.
for pid in $(netstat -ano 2>/dev/null | grep "LISTENING" | grep ":$HTTP_PORT " | awk '{print $NF}' | sort -u); do
  echo "http   : killing stale listener pid $pid on port $HTTP_PORT"
  taskkill //PID "$pid" //F > /dev/null 2>&1 || true
done

"$ADB" forward tcp:$DEV_PORT tcp:$DEV_PORT > /dev/null
echo "forward: 127.0.0.1:$DEV_PORT -> device:$DEV_PORT"
echo "http   : http://127.0.0.1:$HTTP_PORT"

exec python "$(dirname "$0")/serve_http.py" "$HTTP_PORT" "$DEV_PORT"
