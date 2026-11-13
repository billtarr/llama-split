#!/system/bin/sh
# split run, dumping stage A's exported hidden state each time, then md5 it
# usage: dumpcheck.sh <k> <runs>
K=${1:-16}
NR=${2:-3}
P="The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century."
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib
i=1
while [ $i -le $NR ]; do
  timeout 900 ./bin/llama-split -m /data/local/tmp/gguf/qwen35-4b-q4_0.gguf -p "$P" -c 64 -n 1 -t 4 -k $K --embd 2560 --dump /data/local/tmp/hid_$i.bin > /dev/null 2>&1
  cp /data/local/tmp/split-B.log /data/local/tmp/B_$i.log
  echo "run $i: $(md5sum /data/local/tmp/hid_$i.bin)"
  grep -E '^\[B\] top5' /data/local/tmp/B_$i.log
  i=$((i + 1))
done
