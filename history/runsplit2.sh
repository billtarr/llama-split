#!/system/bin/sh
# runsplit2.sh <model> <n_embd> <k> <n_predict> <runs>
M=${1:?model}
E=${2:?n_embd}
K=${3:?k}
NP=${4:-4}
NR=${5:-3}
P="The history of computing began with mechanical calculators in the seventeenth century and continued through the analytical engine, the tabulating machine, and the first electronic computers of the mid twentieth century."
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib
i=1
while [ $i -le $NR ]; do
  timeout 900 ./bin/llama-split -m "$M" -p "$P" -c 64 -n $NP -t 4 -k $K --embd $E > /dev/null 2>&1
  cp /data/local/tmp/split-A.log /data/local/tmp/A_$i.log
  cp /data/local/tmp/split-B.log /data/local/tmp/B_$i.log
  echo "== run $i =="
  grep -E '^\[A\] hidden|^gen' /data/local/tmp/A_$i.log
  grep -E '^\[B\] top5' /data/local/tmp/B_$i.log
  i=$((i + 1))
done
