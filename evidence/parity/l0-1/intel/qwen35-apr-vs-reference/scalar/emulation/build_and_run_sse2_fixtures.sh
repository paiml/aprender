#!/usr/bin/env bash
# PMAT-3091 scalar (intel): ggml_sse2_fixtures.c linked against the SCALAR libggml-cpu/-base (the reference), and the same
# source against the NATIVE build (AVX-512/AVX2 ggml_v_silu) as a discrimination check: the hashes must differ there.
set -uo pipefail
B="$HOME/parity-ref/llama-scalar-d1d3c3396"; N="$HOME/src/llama.cpp-d1d3c3396/build"; S="$HOME/parity-ref/scalar-3091"
cd "$S"
echo "host=$(hostname) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
sha256sum ggml_sse2_fixtures.c "$(readlink -f "$B/bin/libggml-cpu.so")" "$(readlink -f "$N/bin/libggml-cpu.so")"
for v in scalar native; do
  lib="$B/bin"; [ "$v" = native ] && lib="$N/bin"
  nice -n 10 timeout 300 gcc -O2 -ffp-contract=off -std=c11 -Wall -Wextra -o "ggml_sse2_fixtures_$v" ggml_sse2_fixtures.c \
    -L"$lib" -lggml-cpu -lggml-base -lm -Wl,-rpath,"$lib" < /dev/null; echo "build $v rc=$?"
  ldd "./ggml_sse2_fixtures_$v" | grep ggml-cpu
  timeout 300 "./ggml_sse2_fixtures_$v" "sse2_fixtures_$v.rs.txt" < /dev/null > "sse2_fixtures_$v.tsv"; echo "run $v rc=$?"
  sha256sum "sse2_fixtures_$v.tsv" "sse2_fixtures_$v.rs.txt"
done
echo "rows=$(($(wc -l < sse2_fixtures_scalar.tsv) - 1)) out_fnv_differ_scalar_vs_native=$(paste <(cut -f7 sse2_fixtures_scalar.tsv) <(cut -f7 sse2_fixtures_native.tsv) | tail -n +2 | awk '$1 != $2' | wc -l)"
for f in ggml_vec_silu_f32 ggml_vec_swiglu_f32 ggml_vec_soft_max_f32; do
  body=$(objdump -d --no-show-raw-insn "$(readlink -f "$B/bin/libggml-cpu.so")" | awk -v s="<$f>:" '$0 ~ s {on=1; next} on && /^$/ {exit} on')
  echo "scalar sym $f ps_insns=$(printf '%s\n' "$body" | grep -cE '\s[a-z]+ps\s') ymm=$(printf '%s\n' "$body" | grep -c ymm) fma=$(printf '%s\n' "$body" | grep -c 'vfn\?m\(add\|sub\)') calls=$(printf '%s\n' "$body" | grep -o 'call.*<[^>]*>' | sort -u | tr '\n' ';')"
done
echo "end_utc=$(date -u +%FT%TZ)"
