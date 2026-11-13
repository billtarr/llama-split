#!/system/bin/sh
# repeat the single-process reference run N times (same prompt as runsplit.sh)
# usage: runsingle.sh <model> <runs> [extra args...]
M=${1:-/data/local/tmp/gguf/qwen35-4b-q4_0.gguf}
NR=${2:-3}
shift 2
P="The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century."
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib
i=1
while [ $i -le $NR ]; do
  echo "== run $i =="
  timeout 900 ./bin/llama-split -m "$M" -p "$P" -c 64 -n 4 -t 4 "$@" 2>&1 | grep -E '^\[S\] (prefill|top5)|^tokens'
  i=$((i + 1))
done
