#!/usr/bin/env bash
# PMAT-3303 per-token confound: batched regression run (n=1, must reproduce the committed reference
# bytes), per-token runs (n=3), then comparisons A/B/C with the committed PMAT-3091 comparator
# (compare_raw_logits.py, sha256 f05cd87a...d4ce). intel, -t 8, one job at a time.
set -uo pipefail
L="${LLAMA_DIR:-$HOME/src/llama.cpp-d1d3c3396}"
BIN="$L/apr-raw-logits/apr_raw_logits"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"
PROMPT="${PROMPT_FILE:-/tmp/ref3303/prompt.txt}"
CMPPY="${CMPPY:-$HOME/parity-ref/apr-3091-subject/compare_raw_logits.py}"
BATCHED="$HOME/parity-ref/qwen35-0.8b-q4km-d1d3c3396/qwen35-0.8b-q4km-d1d3c3396.rawlogits.bin"
APR="$HOME/parity-ref/apr-3091-subject/apr-intel-run1.bin"
OUT="$HOME/parity-ref/qwen35-0.8b-q4km-d1d3c3396-per-token"
mkdir -p "$OUT"
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) host=$(hostname) nproc=$(nproc)"
sha256sum "$MODEL" "$PROMPT" "$BIN" "$CMPPY" "$BATCHED" "$APR"

run() { # name mode-flag...
  local name="$1"; shift
  echo "run $name start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  timeout 900 "$BIN" -m "$MODEL" -f "$PROMPT" -ngl 0 -t 8 -c 78 -b 78 "$@" \
    --raw-out "$OUT/$name.bin" < /dev/null > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "run $name rc=$rc end_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg) bytes=$(stat -c %s "$OUT/$name.bin" 2>/dev/null)"
  grep -E '^apr_raw_logits:|n_threads' "$OUT/$name.log"
  sha256sum "$OUT/$name.bin"
}

run batched-regress
cmp "$OUT/batched-regress.bin" "$BATCHED"; echo "batched-regress vs committed reference cmp rc=$?"
for i in 1 2 3; do run "per-token-run$i" --per-token; done
for i in 2 3; do cmp "$OUT/per-token-run1.bin" "$OUT/per-token-run$i.bin"; echo "per-token run1 vs run$i cmp rc=$?"; done

compare() { # label ref sub
  echo "== $1: ref=$(basename "$2") sub=$(basename "$3")"
  cmp -s "$2" "$3"; echo "$1 cmp rc=$? (0 = byte-identical)"
  timeout 600 python3 "$CMPPY" "$2" "$3" --json "$OUT/$1.json" < /dev/null > "$OUT/$1.tsv"
  echo "$1 compare rc=$?"
  grep '^summary' "$OUT/$1.tsv"
}
compare A-per-token-vs-batched "$OUT/per-token-run1.bin" "$BATCHED"
compare B-per-token-vs-apr     "$OUT/per-token-run1.bin" "$APR"
compare C-batched-vs-apr       "$BATCHED" "$APR"
sha256sum "$OUT"/*.tsv "$OUT"/*.json
