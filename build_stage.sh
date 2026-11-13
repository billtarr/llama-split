#!/usr/bin/env bash
# build a stage library against the ABI: ./build_stage.sh stage_pass.c  -> stage_pass.so
# It must link nothing from llama.cpp -- that is what makes the stage backend-agnostic.
set -e
cd "$(dirname "$0")"
SRC="${1:-stage_pass.c}"
OUT="${SRC%.c}.so"
export MSYS_NO_PATHCONV=1
HERE="$(pwd -W 2>/dev/null || pwd)"
INNER='set -e; CC=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang; $CC -O2 -fPIC -shared -I/src /src/'"$SRC"' -o /tmp/out.so; base64 -w0 /tmp/out.so'
docker run -i --rm -v "$HERE:/src:ro" \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c "$INNER" \
  < /dev/null | base64 -d > "$OUT"
ls -l "$OUT"
