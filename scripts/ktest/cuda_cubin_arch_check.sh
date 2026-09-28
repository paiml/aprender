#!/usr/bin/env bash
# KTEST-05 / F-4: prove the SASS apr actually RAN was compiled for the device's own arch.
#
# apr ships PTX only (no AOT fatbin; cop ruling). The driver JIT-compiles it and trueno caches the
# result at $HOME/.cache/trueno/ptx/<sha256>.cubin (crates/aprender-gpu/src/driver/ptx_cache.rs).
# So the F-4 subject is that JIT cache, not a fatbin: run apr once with a PRIVATE $HOME, then read
# every cubin's ELF arch with `cuobjdump --list-elf`. On apr the PTX-JIT arm IS the default path,
# so the L2 parity receipts already exercise it; this check proves which SASS that was.
#
# Usage: cuda_cubin_arch_check.sh <apr> <model.gguf> <outdir>
#        cuda_cubin_arch_check.sh --self-test
# Env:   KTEST_EXPECT_SM=sm_NN overrides the arch read from nvidia-smi (the planted F-4 falsifier:
#        a wrong expectation must turn the verdict RED).
# Exit:  0 PASS, 1 RED (mismatch/empty), 2 setup error (no device, no cuobjdump, apr failed).
set -uo pipefail

# classify <expected sm_NN> <file with one arch per line> -> prints verdict, returns 0/1
classify() {
  local want=$1 f=$2 n bad
  n=$(grep -c . "$f" 2>/dev/null) || n=0
  if [ "$n" -eq 0 ]; then echo "RED_EMPTY n=0 expected=$want"; return 1; fi
  bad=$(grep -vxF "$want" "$f" | sort | uniq -c | tr -s ' ' | tr '\n' ';')
  if [ -n "$bad" ]; then echo "RED_MISMATCH n=$n expected=$want other=${bad}"; return 1; fi
  echo "PASS n=$n arch=$want"; return 0
}

# cc_to_sm "12.1" -> sm_121
cc_to_sm() { local cc=${1//[[:space:]]/}; case $cc in [0-9]*.[0-9]*) echo "sm_${cc%%.*}${cc#*.}";; *) return 1;; esac; }

# arch_of_listing: cuobjdump --list-elf output on stdin -> one sm_NN per ELF line
arch_of_listing() { grep -oE '\.sm_[0-9]+[a-z]?\.cubin' | sed -E 's/^\.//; s/\.cubin$//'; }

self_test() {
  local t fails=0 tmp
  tmp=$(mktemp -d) || return 2
  # case table: name | expected | archs (comma list, "" = none) | want verdict prefix
  while IFS='|' read -r name want archs verdict; do
    tr ',' '\n' <<<"$archs" | grep . > "$tmp/a" || true
    got=$(classify "$want" "$tmp/a"); got=${got%% *}
    if [ "$got" = "$verdict" ]; then t=ok; else t=FAIL; fails=$((fails+1)); fi
    printf '%-4s %-28s want=%-12s got=%s\n' "$t" "$name" "$verdict" "$got"
  done <<'CASES'
all-match|sm_121|sm_121,sm_121,sm_121|PASS
one-foreign|sm_121|sm_121,sm_120,sm_121|RED_MISMATCH
all-foreign-sm89-on-gb10|sm_121|sm_89,sm_89|RED_MISMATCH
empty-cache-fails-closed|sm_121||RED_EMPTY
planted-wrong-expectation|sm_120|sm_121|RED_MISMATCH
substring-not-a-match|sm_12|sm_121|RED_MISMATCH
sm89-ok|sm_89|sm_89|PASS
CASES
  while IFS='|' read -r in want; do
    got=$(cc_to_sm "$in" || echo ERR)
    if [ "$got" = "$want" ]; then t=ok; else t=FAIL; fails=$((fails+1)); fi
    printf '%-4s cc_to_sm %-8s want=%-8s got=%s\n' "$t" "'$in'" "$want" "$got"
  done <<'CASES'
12.1|sm_121
 8.9|sm_89
10.0|sm_100
N/A|ERR
CASES
  got=$(printf 'ELF file    1: 3f2a.sm_121.cubin\nELF file    2: 99.sm_90a.cubin\nPTX file 1: x.ptx\n' | arch_of_listing | tr '\n' ,)
  if [ "$got" = "sm_121,sm_90a," ]; then t=ok; else t=FAIL; fails=$((fails+1)); fi
  echo "$t   arch_of_listing got=$got"
  rm -rf "${tmp:?}"
  echo "self-test: $fails failing case(s)"; [ "$fails" -eq 0 ]
}

[ "${1:-}" = --self-test ] && { self_test; exit $?; }
[ $# -eq 3 ] || { echo "usage: $0 <apr> <model> <outdir> | --self-test" >&2; exit 2; }
APR=$1; M=$2; O=$3
command -v cuobjdump >/dev/null || { echo "SETUP: no cuobjdump" >&2; exit 2; }
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
dev=$(cc_to_sm "$cc") || { echo "RED_NO_DEVICE cc='$cc'"; exit 2; }
want=${KTEST_EXPECT_SM:-$dev}
mkdir -p "$O/home" || exit 2
H=$(cd "$O/home" && pwd)
[ -z "$(find "$H" -name '*.cubin' -print -quit)" ] || { echo "SETUP: $H already holds cubins; use a fresh outdir" >&2; exit 2; }
{ "$APR" --version; sha256sum "$APR" | cut -c1-16; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader; } > "$O/version.txt" 2>&1
HOME=$H BATCHED_PREFILL=1 timeout 600 "$APR" run "$M" --prompt "What is 2+2? Answer with one number." \
  --max-tokens 4 --temperature 0 > "$O/run.out" 2> "$O/run.err"; rc=$?
[ "$rc" -eq 0 ] || { echo "SETUP: apr run rc=$rc (see $O/run.err)" >&2; exit 2; }
: > "$O/archs.txt"
find "$H" -name '*.cubin' -print0 | while IFS= read -r -d '' c; do
  cuobjdump --list-elf "$c" 2>/dev/null | arch_of_listing
done > "$O/archs.txt"
v=$(classify "$want" "$O/archs.txt"); vrc=$?
printf 'device=%s expected=%s apr=%s\n%s\n' "$dev" "$want" "$(head -1 "$O/version.txt")" "$v" | tee "$O/verdict.txt"
exit "$vrc"
