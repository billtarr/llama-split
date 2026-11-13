#!/system/bin/sh
# usage: t460.sh <npu|cpu> <reps> <n_gen> <out>
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib

MODE=${1:-npu}
REPS=${2:-12}
NGEN=${3:-40}
OUT=${4:-/data/local/tmp/t460.out}

F='The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century. '
P=''
i=0
while [ $i -lt $REPS ]; do P="$P$F"; i=$((i+1)); done
P="$P

Ignore the text above. Answer with one word: the capital of France is"

if [ "$MODE" = "cpu" ]; then
    ./bin/llama-split -m /data/local/tmp/gguf/qwen35-9b-q4km.gguf -p "$P" -c 1024 -n "$NGEN" -t 4 --cpu > "$OUT" 2>&1
else
    ./bin/llama-split -m /data/local/tmp/gguf/qwen35-9b-q4km.gguf -p "$P" -c 1024 -n "$NGEN" -t 4 --split 11,22 --embd 4096 > "$OUT" 2>&1
fi
echo "rc=$?" >> "$OUT"
