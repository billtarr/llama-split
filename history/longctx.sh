#!/system/bin/sh
# longctx.sh - does the split still hold at ctx 1024? (~600-token prompt)
M=/data/local/tmp/gguf/qwen35-4b-q4_0.gguf
P1="The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century."
P2="Each generation reduced the cost of arithmetic and increased the scale of what could be automated, from room sized mainframes to single chip microprocessors and finally to the mobile devices we carry today."
P3="Along the way, software abstractions accumulated, compilers replaced hand written machine code, operating systems multiplexed scarce hardware, and networks turned isolated machines into a shared fabric of computation."
P="$P1 $P2 $P3 $P1 $P2 $P3"
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib

echo "########## single process, ctx 1024 (expected to fail)"
timeout 600 ./bin/llama-split -m $M -p "$P" -c 1024 -n 4 -t 4 2>&1 | grep -E '^\[S\] prompt|^\[S\] prefill|fastrpc_mmap failed' | head -3

echo "########## split 16/16, ctx 1024"
i=1
while [ $i -le 2 ]; do
  timeout 900 ./bin/llama-split -m $M -p "$P" -c 1024 -n 8 -t 4 -k 16 --embd 2560 > /dev/null 2>&1
  echo "== run $i =="
  grep -E '^\[A\] prompt|^\[A\] prefill|^\[A\] decode|^gen' /data/local/tmp/split-A.log
  grep -E '^\[B\] prefill|^\[B\] top5' /data/local/tmp/split-B.log
  i=$((i + 1))
done
