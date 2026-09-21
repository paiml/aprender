#!/usr/bin/env bash
# PMAT-3091 flip variation (intel, one job at a time). Phase T: apr thread confound on the ORIGINAL
# 78 ids (RAYON_NUM_THREADS=8, =1, default), thread count sampled from /proc/<pid>/status.
# Phase P: 4 new prompts x (llama batched, llama --per-token, apr). Phase C: comparisons A/B/C per
# prompt with the committed comparator compare_raw_logits.py (sha256 f05cd87a...d4ce).
set -uo pipefail
L="$HOME/src/llama.cpp-d1d3c3396"
PROD="$L/apr-raw-logits/apr_raw_logits"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"
OUT="$HOME/parity-ref/variation-3091"
APR="$OUT/qwen35_raw_logits"             # built from #3114 31448f6c3
CMPPY="$HOME/parity-ref/apr-3091-subject/compare_raw_logits.py"
IDS0="$HOME/parity-ref/apr-3091-subject/prompt_token_ids.txt"
PDIR="/tmp/flipvar3091"
cd "$OUT" || exit 2
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) host=$(hostname) nproc=$(nproc) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
df -h "$OUT" | tail -1
sha256sum "$MODEL" "$PROD" "$APR" "$CMPPY" "$IDS0" "$PDIR"/prompt-?.txt "$PDIR"/prompt-?.ids

run_apr() { # name ids threads(""=default)
  local name="$1" ids="$2" th="$3" pid maxth=0 t
  echo "apr $name threads=${th:-default} start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  if [ -n "$th" ]; then
    RAYON_NUM_THREADS="$th" /usr/bin/time -v timeout 1800 "$APR" "$MODEL" "$ids" "$OUT/$name.bin" < /dev/null > "$OUT/$name.log" 2>&1 &
  else
    /usr/bin/time -v timeout 1800 "$APR" "$MODEL" "$ids" "$OUT/$name.bin" < /dev/null > "$OUT/$name.log" 2>&1 &
  fi
  local tpid=$!
  while kill -0 "$tpid" 2>/dev/null; do
    pid=$(pgrep -x qwen35_raw_logi | head -1)
    if [ -n "$pid" ]; then
      t=$(awk '/^Threads:/{print $2}' "/proc/$pid/status" 2>/dev/null)
      [ -n "$t" ] && [ "$t" -gt "$maxth" ] && maxth=$t
    fi
    sleep 0.2
  done
  wait "$tpid"; local rc=$?
  echo "apr $name rc=$rc end_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg) max_threads_sampled=$maxth $(grep -E 'Percent of CPU|Elapsed' "$OUT/$name.log" | tr -s ' \t' ' ' | tr '\n' ';')"
  sha256sum "$OUT/$name.bin"
}

run_llama() { # name prompt n mode-flag...
  local name="$1" prompt="$2" n="$3"; shift 3
  echo "llama $name start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  timeout 900 "$PROD" -m "$MODEL" -f "$prompt" -ngl 0 -t 8 -c "$n" -b "$n" "$@" --raw-out "$OUT/$name.bin" < /dev/null > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "llama $name rc=$rc end_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  grep -E '^apr_raw_logits: n_pos' "$OUT/$name.log"
  sha256sum "$OUT/$name.bin"
}

compare() { # label ref sub
  cmp -s "$2" "$3"; echo "$1 cmp rc=$? (0 = byte-identical)"
  timeout 600 python3 "$CMPPY" "$2" "$3" --json "$OUT/$1.json" < /dev/null > "$OUT/$1.tsv"
  echo "$1 compare rc=$?"
  grep '^summary' "$OUT/$1.tsv"
}

echo "== phase T (thread confound, original 78 ids; recorded subject f6f79264...0252)"
run_apr t0-default "$IDS0" ""
run_apr t0-rayon8 "$IDS0" 8
run_apr t0-rayon1 "$IDS0" 1

for k in 1 2 3 4; do
  n=$(tr ',' '\n' < "$PDIR/prompt-$k.ids" | grep -c .)
  echo "== phase P prompt-$k n_tokens=$n"
  run_llama "p$k-batched" "$PDIR/prompt-$k.txt" "$n"
  run_llama "p$k-per-token" "$PDIR/prompt-$k.txt" "$n" --per-token
  run_apr "p$k-apr" "$PDIR/prompt-$k.ids" ""
done

echo "== phase C"
for k in 1 2 3 4; do
  compare "p$k-A-per-token-vs-batched" "$OUT/p$k-per-token.bin" "$OUT/p$k-batched.bin"
  compare "p$k-B-per-token-vs-apr" "$OUT/p$k-per-token.bin" "$OUT/p$k-apr.bin"
  compare "p$k-C-batched-vs-apr" "$OUT/p$k-batched.bin" "$OUT/p$k-apr.bin"
done
sha256sum "$OUT"/*.tsv "$OUT"/*.json
echo "done end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
