#!/usr/bin/env bash
# Build apr_raw_logits + raw_logits_compare against the llama.cpp d1d3c3396 CPU build (intel).
# The llama.cpp tree must already be built (build/bin has libllama.so, libllama-common.so).
set -euo pipefail
L="${LLAMA_DIR:-$HOME/src/llama.cpp-d1d3c3396}"
SRC="${SRC_DIR:-$(cd "$(dirname "$0")" && pwd)}"
OUT="$L/apr-raw-logits"
test "$(git -C "$L" rev-parse --short=9 HEAD)" = "d1d3c3396" || { echo "llama.cpp tree is not d1d3c3396" >&2; exit 2; }
mkdir -p "$OUT"
c++ -O2 -std=c++17 -Wall -Wextra \
  -I"$L/include" -I"$L/common" -I"$L/ggml/include" \
  "$SRC/apr_raw_logits.cpp" -o "$OUT/apr_raw_logits" \
  -L"$L/build/bin" -lllama-common -lllama -lggml -lggml-base \
  -Wl,-rpath,"$L/build/bin"
c++ -O2 -std=c++17 -Wall -Wextra "$SRC/raw_logits_compare.cpp" -o "$OUT/raw_logits_compare"
sha256sum "$OUT/apr_raw_logits" "$OUT/raw_logits_compare"
