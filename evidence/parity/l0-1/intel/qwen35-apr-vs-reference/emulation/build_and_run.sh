#!/usr/bin/env bash
# PMAT-3091: build ggml_emul_fixtures.c against the intel llama.cpp d1d3c3396 ggml shared libs and run it.
# Host writes: ~/parity-ref/emulation-3091/ only. Also records whether the GENERIC dot symbols in the shipped
# libggml-cpu.so contain FMA instructions (compiler contraction would make a no-FMA port differ by an ulp).
set -euo pipefail
L="${LLAMA_DIR:-$HOME/src/llama.cpp-d1d3c3396}"
W="${WORK:-$HOME/parity-ref/emulation-3091}"
mkdir -p "$W"
cd "$W"
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) host=$(hostname) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
gcc --version | head -1
sha256sum ggml_emul_fixtures.c "$L/build/bin/libggml-cpu.so.0.24.0" "$L/build/bin/libggml-base.so.0.24.0"
timeout 300 gcc -O2 -ffp-contract=off -std=c11 -Wall -Wextra -o ggml_emul_fixtures ggml_emul_fixtures.c \
  -L"$L/build/bin" -lggml-cpu -lggml-base -Wl,-rpath,"$L/build/bin" < /dev/null
timeout 300 ./ggml_emul_fixtures fixtures.rs.txt fixtures.bin < /dev/null > fixtures.tsv
echo "fixtures rc=$?"
sha256sum fixtures.tsv fixtures.rs.txt fixtures.bin
for f in ggml_vec_dot_q4_K_q8_K_generic ggml_vec_dot_q5_K_q8_K_generic ggml_vec_dot_q6_K_q8_K_generic ggml_vec_dot_q8_0_q8_0_generic quantize_row_q8_K_ref; do
  lib="$L/build/bin/libggml-cpu.so.0.24.0"
  [ "$f" = quantize_row_q8_K_ref ] && lib="$L/build/bin/libggml-base.so.0.24.0"
  n=$(objdump -d --no-show-raw-insn "$lib" | awk -v s="<$f>:" '$0 ~ s {on=1; next} on && /^$/ {exit} on' | grep -c 'vfn\?m\(add\|sub\)' || true)
  echo "fma_insns $f $n"
done
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
