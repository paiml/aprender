#!/usr/bin/env bash
# PMAT-3091 scalar: the SAME ggml_emul_fixtures.c (emulation/, unchanged) re-linked against the SCALAR libggml-cpu/-base
# (~/parity-ref/llama-scalar-d1d3c3396/bin). Output = fixtures_scalar.{rs.txt,tsv,bin}. Also counts FMA and ymm/zmm
# instructions in the generic dot / quantize symbols and in the whole scalar libggml-cpu (engaged proof).
set -uo pipefail
L="$HOME/src/llama.cpp-d1d3c3396"; B="$HOME/parity-ref/llama-scalar-d1d3c3396"; S="$HOME/parity-ref/scalar-3091"
cd "$S"
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) host=$(hostname) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
CPU="$(readlink -f "$B/bin/libggml-cpu.so")"; BASE="$(readlink -f "$B/bin/libggml-base.so")"
sha256sum ggml_emul_fixtures.c "$CPU" "$BASE"
timeout 300 gcc -O2 -ffp-contract=off -std=c11 -Wall -Wextra -o ggml_emul_fixtures_scalar ggml_emul_fixtures.c \
  -L"$B/bin" -lggml-cpu -lggml-base -Wl,-rpath,"$B/bin" < /dev/null; echo "harness build rc=$?"
ldd ./ggml_emul_fixtures_scalar | grep ggml
timeout 300 ./ggml_emul_fixtures_scalar fixtures_scalar.rs.txt fixtures_scalar.bin < /dev/null > fixtures_scalar.tsv; echo "fixtures rc=$?"
sha256sum fixtures_scalar.tsv fixtures_scalar.rs.txt fixtures_scalar.bin
for f in ggml_vec_dot_q4_K_q8_K_generic ggml_vec_dot_q5_K_q8_K_generic ggml_vec_dot_q6_K_q8_K_generic ggml_vec_dot_q8_0_q8_0_generic \
         ggml_vec_dot_q4_K_q8_K ggml_vec_dot_q8_0_q8_0 quantize_row_q8_0 quantize_row_q8_K_ref quantize_row_q8_0_ref; do
  lib="$CPU"; case "$f" in quantize_row_q8_K_ref|quantize_row_q8_0_ref) lib="$BASE";; esac
  body=$(objdump -d --no-show-raw-insn "$lib" | awk -v s="<$f>:" '$0 ~ s {on=1; next} on && /^$/ {exit} on')
  echo "sym $f lines=$(printf '%s\n' "$body" | grep -c . ) fma=$(printf '%s\n' "$body" | grep -c 'vfn\?m\(add\|sub\)') ymm=$(printf '%s\n' "$body" | grep -c ymm) calls=$(printf '%s\n' "$body" | grep -o 'call.*<[^>]*>' | tr '\n' ';')"
done
for lib in "$CPU" "$BASE" "$(readlink -f "$B/bin/libllama.so")"; do
  d=$(objdump -d --no-show-raw-insn "$lib")
  echo "lib ${lib##*/} insns=$(printf '%s\n' "$d" | grep -c '^ *[0-9a-f]*:') ymm=$(printf '%s\n' "$d" | grep -c ymm) zmm=$(printf '%s\n' "$d" | grep -c zmm) fma=$(printf '%s\n' "$d" | grep -c 'vfn\?m\(add\|sub\)') vex_v_mnemonics=$(printf '%s\n' "$d" | grep -cE '\s(v[a-z]+(ps|pd|ss|sd)|vp[a-z]+)\s')"
done
echo "--- compile flags of ggml-cpu sources (compile_commands.json)"
python3 - "$B/compile_commands.json" <<'PY'
import json, sys, re
cc = json.load(open(sys.argv[1]))
for e in cc:
    f = e["file"]
    if re.search(r"ggml-cpu/(quants\.c|ggml-cpu\.c|ops\.cpp|vec\.cpp|arch/x86/quants\.c)$", f):
        cmd = e.get("command") or " ".join(e["arguments"])
        flags = [t for t in cmd.split() if t.startswith(("-O", "-m", "-f", "-D GGML", "-DGGML", "-std"))]
        print(f.split("llama.cpp-d1d3c3396/")[-1], " ".join(flags))
PY
echo "gcc default fp-contract (C, GNU mode): $(gcc -Q --help=optimizers -O3 2>/dev/null | grep -E 'ffp-contract' | tr -s ' ')"
echo "end_utc=$(date -u +%FT%TZ)"
