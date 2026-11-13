#!/system/bin/sh
# does a second resident DSP process make a single-process run non-deterministic?
# 1) start a small model in the background (holds a second PD/session)
# 2) run the 4B single-process reference 3x and compare top5
M4=/data/local/tmp/gguf/qwen35-4b-q4_0.gguf
M08=/data/local/tmp/gguf/qwen35-0.8b-q8.gguf
P="The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century."
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib

echo "--- baseline (nothing else running) ---"
timeout 300 ./bin/llama-split -m $M4 -p "$P" -c 64 -n 1 -t 4 2>&1 | grep -E '^\[S\] top5'

echo "--- start background 0.8b session ---"
timeout 400 ./bin/llama-split -m $M08 -p "$P" -c 128 -n 100 -t 4 > /data/local/tmp/bg08.log 2>&1 &
BGPID=$!
sleep 12

echo "--- with second session resident ---"
i=1
while [ $i -le 3 ]; do
  timeout 300 ./bin/llama-split -m $M4 -p "$P" -c 64 -n 1 -t 4 2>&1 | grep -E '^\[S\] top5|fastrpc_mmap failed'
  i=$((i + 1))
done

kill $BGPID 2>/dev/null
wait $BGPID 2>/dev/null
echo "--- background log ---"
grep -E 'decode:|prefill ' /data/local/tmp/bg08.log | tail -3
