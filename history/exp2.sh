#!/bin/sh
# production config check: ctx 8192 (KV x2 slots) + lm_head on the DSP
cd /data/local/tmp/llama.cpp
M=/data/local/tmp/gguf/qwen35-9b-q4km.gguf
BIN=./bin/llama-split
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
clean() { for p in $(ps -A | grep '[l]lama-split' | awk '{print $2}'); do kill -9 $p; done; sleep 1; }

probe() {
  clean; rm -f /data/local/tmp/split-*.log
  SPLIT_OUTPUT_HTP=1 $BIN -m $M -p 'hi' -c 8192 -n 2 -t 4 \
      --split $SP --embd 4096 --batch 64 --slots 2 2>&1 | grep -aE 'stages:|status|failed' | tail -2
  bad=$(grep -al 'fastrpc_mmap failed\|ctx init failed\|abort' /data/local/tmp/split-*.log 2>/dev/null | head -2)
  printf "  -> %s\n" "${bad:-OK}"
}

echo "=== 11,22 ctx8192 x2 slots (C=10 layers + lm_head + KV)"; SP=11,22; probe
echo "=== 11,23 ctx8192 x2 slots (C=9)";                        SP=11,23; probe
echo "=== 12,23 ctx8192 x2 slots (A=12)";                       SP=12,23; probe
clean
