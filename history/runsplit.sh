#!/system/bin/sh
# repeat the split run N times, keeping per-run A/B logs
# usage: runsplit.sh <k> <n_predict> <runs>
K=${1:-16}
NP=${2:-4}
NR=${3:-3}
P="The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century."
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib
i=1
while [ $i -le $NR ]; do
  timeout 900 ./bin/llama-split -m /data/local/tmp/gguf/qwen35-4b-q4_0.gguf -p "$P" -c 64 -n $NP -t 4 -k $K --embd 2560 > /dev/null 2>&1
  cp /data/local/tmp/split-A.log /data/local/tmp/A_$i.log
  cp /data/local/tmp/split-B.log /data/local/tmp/B_$i.log
  echo "== run $i =="
  grep -E '^\[A\] (hidden|prefill|decode)|^gen' /data/local/tmp/A_$i.log
  grep -E '^\[B\] (prefill|top5)' /data/local/tmp/B_$i.log
  i=$((i + 1))
done
