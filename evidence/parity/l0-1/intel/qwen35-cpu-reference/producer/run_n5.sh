#!/usr/bin/env bash
# n=5 determinism runs of apr_raw_logits on intel (-t 8, one job at a time), then the falsifier.
set -uo pipefail
L="${LLAMA_DIR:-$HOME/src/llama.cpp-d1d3c3396}"
BIN="$L/apr-raw-logits/apr_raw_logits"
CMP="$L/apr-raw-logits/raw_logits_compare"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"
PROMPT="${PROMPT_FILE:-/tmp/ref3303/prompt.txt}"
KLD="${KLD_FILE:-/tmp/ref3303/run1.kld}"
OUT="$HOME/parity-ref/qwen35-0.8b-q4km-d1d3c3396"
mkdir -p "$OUT"
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) host=$(hostname)"
sha256sum "$MODEL" "$PROMPT" "$BIN"
for i in 1 2 3 4 5; do
  echo "run $i start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  timeout 600 "$BIN" -m "$MODEL" -f "$PROMPT" -ngl 0 -t 8 -c 78 -b 78 \
    --raw-out "$OUT/raw-run$i.bin" < /dev/null > "$OUT/raw-run$i.log" 2>&1
  rc=$?
  echo "run $i rc=$rc end_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg) bytes=$(stat -c %s "$OUT/raw-run$i.bin" 2>/dev/null)"
  grep -E '^apr_raw_logits:|n_threads' "$OUT/raw-run$i.log"
  sha256sum "$OUT/raw-run$i.bin"
done
echo "falsifier:"
timeout 600 "$CMP" "$OUT/raw-run1.bin" "$KLD" < /dev/null
echo "compare rc=$?"
