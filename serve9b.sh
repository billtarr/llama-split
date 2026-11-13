#!/system/bin/sh
# three-stage layer-split server for Qwen3.5-9B Q4_K_M (32 layers -> 11 / 13 / 8)
#
# v2 pipeline is the default: --slots 2 gives two independent contexts per stage so stage 0 can
# push request B while request A waits for its token (1.15~1.4x aggregate throughput, see L-055).
#
# The last stage is deliberately small (8 layers, not 10) so `output.weight` (Q6_K, 796 MiB) fits
# on the DSP next to it. Keeping the lm_head on the DSP is worth -13% latency and it also stops
# the CPU matmul from slowing every *other* stage down (L-058). With 9 layers a single request
# still works but two concurrent ones blow the VA window; 8 layers is the safe point.
#
# IMPORTANT: slots>1 forces --batch 64. With batch 256 the second context cannot initialise at
# all (52 MiB of compute buffers per context x2 does not fit in the 2048 MiB VA window).
# Override with env: BATCH=... SLOTS=... CTX=... SPLIT=... ROUTE=...
cd /data/local/tmp/llama.cpp || exit 1
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib
exec ./bin/llama-split \
    -m /data/local/tmp/gguf/qwen35-9b-q4km.gguf \
    -c ${CTX:-8192} --split ${SPLIT:-11,24} --embd 4096 \
    --batch ${BATCH:-64} --slots ${SLOTS:-2} \
    --route "${ROUTE:-.*output\.weight=htp}" \
    --serve ${1:-18080} ${EXTRA}
