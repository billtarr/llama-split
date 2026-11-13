#!/system/bin/sh
# injectcheck.sh <model> <n_embd> <k> <runs>
M=${1:?model}
E=${2:?n_embd}
K=${3:?k}
NR=${4:-3}
P="The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century."
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib

# 1) full split run, dumping stage A's hidden state
timeout 900 ./bin/llama-split -m "$M" -p "$P" -c 64 -n 1 -t 4 -k $K --embd $E --dump /data/local/tmp/inj.bin > /dev/null 2>&1
echo "hidden: $(md5sum /data/local/tmp/inj.bin)"

# 2) stage B alone, 3x, on that exact input
i=1
while [ $i -le $NR ]; do
  echo "== inject run $i =="
  timeout 900 ./bin/llama-split -m "$M" -c 64 -t 4 -k $K --embd $E --inject /data/local/tmp/inj.bin 2>&1 | grep -E '^\[B\] (prefill|top5)|^\[M\] inject'
  i=$((i + 1))
done
