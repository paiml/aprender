#!/usr/bin/env bash
# PMAT-3091 layerwise: llama.cpp d1d3c3396 per-layer dumps via the scheduler eval callback.
# 1. regression: --per-token without dumps must reproduce the committed per-token logits bytes
#    (p4 0eca077c..., original 2401c110...) after the producer rebuild;
# 2. dump runs: prompt-4 pos 0-3, original prompt pos 4 and 28; their logits must ALSO equal the
#    per-token bytes (proof the callback did not alter the computation);
# 3. sha256 over every dumped .f32, appended to the manifest.
# intel, -t 8, one job at a time, timeout + </dev/null on every binary.
set -uo pipefail
L="${LLAMA_DIR:-$HOME/src/llama.cpp-d1d3c3396}"
BIN="$L/apr-raw-logits/apr_raw_logits"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"
W="${WORK:-/tmp/layer3091}"
OUT="${OUT:-$HOME/parity-ref/layerwise-3091}"
P0="${P0:-/tmp/ref3303/prompt.txt}"
P4="${P4:-$W/prompt-4.txt}"
REF_P0="$HOME/parity-ref/qwen35-0.8b-q4km-d1d3c3396-per-token/per-token-run1.bin"
REF_P4="$HOME/parity-ref/variation-3091/p4-per-token.bin"
mkdir -p "$OUT"
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) host=$(hostname) nproc=$(nproc) start_utc=$(date -u +%FT%TZ)"
sha256sum "$MODEL" "$BIN" "$P0" "$P4" "$REF_P0" "$REF_P4"

run() { # name prompt n extra...
  local name="$1" prompt="$2" n="$3"; shift 3
  echo "run $name start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  timeout 900 "$BIN" -m "$MODEL" -f "$prompt" -ngl 0 -t 8 -c "$n" -b "$n" --per-token "$@" \
    --raw-out "$OUT/$name.bin" < /dev/null > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "run $name rc=$rc end_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  grep -E '^apr_raw_logits:' "$OUT/$name.log"
  sha256sum "$OUT/$name.bin"
}

run regress-p4 "$P4" 82
cmp "$OUT/regress-p4.bin" "$REF_P4"; echo "regress-p4 vs variation p4-per-token cmp rc=$?"
run regress-p0 "$P0" 78
cmp "$OUT/regress-p0.bin" "$REF_P0"; echo "regress-p0 vs per-token-run1 cmp rc=$?"

rm -rf "$OUT/dump-p4" "$OUT/dump-p0"
run dump-p4 "$P4" 82 --dump-tensors "$OUT/dump-p4" --dump-positions 0,1,2,3
cmp "$OUT/dump-p4.bin" "$REF_P4"; echo "dump-p4 logits vs variation p4-per-token cmp rc=$?"
run dump-p0 "$P0" 78 --dump-tensors "$OUT/dump-p0" --dump-positions 4,28
cmp "$OUT/dump-p0.bin" "$REF_P0"; echo "dump-p0 logits vs per-token-run1 cmp rc=$?"

for d in dump-p4 dump-p0; do
  ( cd "$OUT/$d" && { head -1 manifest.tsv | tr -d '\n'; printf '\tsha256\n'; tail -n +2 manifest.tsv | while IFS=$'\t' read -r nm il pos shp nb f; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$nm" "$il" "$pos" "$shp" "$nb" "$f" "$(sha256sum "$f" | cut -d' ' -f1)"; done; } > manifest.sha256.tsv )
  echo "$d manifest rows=$(($(wc -l < "$OUT/$d/manifest.sha256.tsv") - 1)) names=$(($(wc -l < "$OUT/$d/tensor_names.tsv") - 1))"
  sha256sum "$OUT/$d/tensor_names.tsv" "$OUT/$d/manifest.sha256.tsv"
done
cmp "$OUT/dump-p4/tensor_names.tsv" "$OUT/dump-p0/tensor_names.tsv"; echo "tensor_names p4 vs p0 cmp rc=$?"
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
