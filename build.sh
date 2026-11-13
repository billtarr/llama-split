#!/usr/bin/env bash
# build llama-split and emit the binary on stdout as base64 (avoids docker bind-mount writeback races).
# The sources come from a read-only mount so stage_abi.h resolves without piping two files.
set -e
cd "$(dirname "$0")"
export MSYS_NO_PATHCONV=1
HERE="$(pwd -W 2>/dev/null || pwd)"
INNER='set -e; CC=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++; $CC -O2 -std=c++17 /src/llama-split.cpp -I/src -I/pkg/include -L/pkg/lib -lllama -lggml -lggml-base -ldl -o /tmp/llama-split; base64 -w0 /tmp/llama-split'
# LLAMA_CPP must point at a llama.cpp checkout that already has patches/llama-layer-split.patch
# applied and was built for arm64-android (headers in include/, libs in lib/).
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
docker run -i --rm -v "$LLAMA_CPP:/pkg:ro" -v "$HERE:/src:ro" \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c "$INNER" \
  < /dev/null | base64 -d > llama-split.new
mv llama-split.new llama-split
ls -l llama-split
