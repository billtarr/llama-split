#!/system/bin/sh
# bench4b.sh - pp/tg for the 4B q4_0, single-process vs split, plus the ctx=256 case
M=/data/local/tmp/gguf/qwen35-4b-q4_0.gguf
SHORT="The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century."
LONG="The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century. Each generation reduced the cost of arithmetic and increased the scale of what could be automated, from room sized mainframes to single chip microprocessors and finally to the mobile devices we carry today. Along the way, software abstractions accumulated, compilers replaced hand written machine code, operating systems multiplexed scarce hardware, and networks turned isolated machines into a shared fabric of computation."
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib

R=${1:-2}

echo "########## A. single process, ctx 64, short prompt"
i=1
while [ $i -le $R ]; do
  timeout 600 ./bin/llama-split -m $M -p "$SHORT" -c 64 -n 16 -t 4 2>&1 | grep -E '^\[S\] prefill|^\[S\] top5'
  i=$((i + 1))
done

echo "########## B. split 16/16, ctx 64, short prompt"
i=1
while [ $i -le $R ]; do
  timeout 900 ./bin/llama-split -m $M -p "$SHORT" -c 64 -n 16 -t 4 -k 16 --embd 2560 > /dev/null 2>&1
  grep -E '^\[A\] prefill|^\[A\] decode' /data/local/tmp/split-A.log
  grep -E '^\[B\] prefill|^\[B\] top5' /data/local/tmp/split-B.log
  i=$((i + 1))
done

echo "########## C. single process, ctx 256, long prompt (expected to fail)"
timeout 600 ./bin/llama-split -m $M -p "$LONG" -c 256 -n 4 -t 4 2>&1 | grep -E '^\[S\] prefill|^\[S\] top5|fastrpc_mmap failed' | head -3

echo "########## D. split 16/16, ctx 256, long prompt"
i=1
while [ $i -le $R ]; do
  timeout 900 ./bin/llama-split -m $M -p "$LONG" -c 256 -n 16 -t 4 -k 16 --embd 2560 > /dev/null 2>&1
  grep -E '^\[A\] prefill|^\[A\] decode' /data/local/tmp/split-A.log
  grep -E '^\[B\] prefill|^\[B\] top5' /data/local/tmp/split-B.log
  i=$((i + 1))
done
