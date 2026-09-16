#!/usr/bin/env bash
# PMAT-3091 layer observer: llama.cpp d1d3c3396 SUB-LAYER dumps at the points apr's
# forward_single_qwen35_observed emits (layerwise/observer_mapping.tsv).
# 1. regression: the producer is NOT rebuilt (sha256 must still be 5c5df52d...); --per-token without dumps
#    must reproduce the committed per-token logits bytes (p4 0eca077c..., original 2401c110...);
# 2. dump runs at p4 pos 0-3 and original pos 4,28 with the sub-layer regex; their logits must ALSO equal
#    the per-token bytes (the callback did not alter the computation);
# 3. sha256 over every dumped .f32.
# intel, -t 8, one job at a time, timeout + </dev/null on every binary.
set -uo pipefail
L="${LLAMA_DIR:-$HOME/src/llama.cpp-d1d3c3396}"
BIN="$L/apr-raw-logits/apr_raw_logits"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"
OUT="${OUT:-$HOME/parity-ref/layerwise-3091}"
P0="${P0:-/tmp/ref3303/prompt.txt}"
P4="${P4:-/tmp/layer3091/prompt-4.txt}"
REF_P0="$HOME/parity-ref/qwen35-0.8b-q4km-d1d3c3396-per-token/per-token-run1.bin"
REF_P4="$HOME/parity-ref/variation-3091/p4-per-token.bin"
RE='model\.input_embed|(attn_norm|conv_output_silu|q_conv_predelta|k_conv_predelta|v_conv_predelta|a_softplus|gate|beta_sigmoid|z|state_predelta|attn_output|new_state|final_output|linear_attn_out|attn_pregate|attn_gated|attn_residual|attn_post_norm|ffn_out|l_out)-[0-9]+|result_norm|result_output'
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) host=$(hostname) nproc=$(nproc) start_utc=$(date -u +%FT%TZ)"
echo "regex=$RE"
sha256sum "$MODEL" "$BIN" "$P0" "$P4" "$REF_P0" "$REF_P4"

run() { # name prompt n extra...
  local name="$1" prompt="$2" n="$3"; shift 3
  echo "run $name start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  timeout 900 "$BIN" -m "$MODEL" -f "$prompt" -ngl 0 -t 8 -c "$n" -b "$n" --per-token "$@" \
    --raw-out "$OUT/$name.bin" < /dev/null > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "run $name rc=$rc end_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  grep -E '^apr_raw_logits:' "$OUT/$name.log" | grep -v token_ids
  grep -c 'not dumped' "$OUT/$name.log" | sed 's/^/not_dumped_lines=/'
  sha256sum "$OUT/$name.bin"
}

run sub-regress-p4 "$P4" 82
cmp "$OUT/sub-regress-p4.bin" "$REF_P4"; echo "sub-regress-p4 vs variation p4-per-token cmp rc=$?"
run sub-regress-p0 "$P0" 78
cmp "$OUT/sub-regress-p0.bin" "$REF_P0"; echo "sub-regress-p0 vs per-token-run1 cmp rc=$?"

rm -rf "${OUT:?}/sub-p4" "${OUT:?}/sub-p0"
run sub-p4 "$P4" 82 --dump-tensors "$OUT/sub-p4" --dump-positions 0,1,2,3 --dump-regex "$RE"
cmp "$OUT/sub-p4.bin" "$REF_P4"; echo "sub-p4 logits vs variation p4-per-token cmp rc=$?"
run sub-p0 "$P0" 78 --dump-tensors "$OUT/sub-p0" --dump-positions 4,28 --dump-regex "$RE"
cmp "$OUT/sub-p0.bin" "$REF_P0"; echo "sub-p0 logits vs per-token-run1 cmp rc=$?"

for d in sub-p4 sub-p0; do
  ( cd "$OUT/$d" && { head -1 manifest.tsv | tr -d '\n'; printf '\tsha256\n'; tail -n +2 manifest.tsv | while IFS=$'\t' read -r nm il pos shp nb f; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$nm" "$il" "$pos" "$shp" "$nb" "$f" "$(sha256sum "$f" | cut -d' ' -f1)"; done; } > manifest.sha256.tsv )
  echo "$d manifest rows=$(($(wc -l < "$OUT/$d/manifest.sha256.tsv") - 1)) names=$(($(wc -l < "$OUT/$d/tensor_names.tsv") - 1))"
  sha256sum "$OUT/$d/tensor_names.tsv" "$OUT/$d/manifest.sha256.tsv"
done
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
