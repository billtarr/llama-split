#!/system/bin/sh
# two-process layer-split server, 8k context, 4k max output
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib
exec ./bin/llama-split \
    -m /data/local/tmp/gguf/qwen35-4b-q4_0.gguf \
    -c 8192 -k 16 --embd 2560 --batch 256 \
    --serve ${1:-18080}
