#!/usr/bin/env bash
# PMAT-3091 kvconfig (intel). llama.cpp d1d3c3396 reference with exact attention arithmetic.
# 0. rebuild the producer (build.sh) with --kv-type/--flash-attn; no-flag per-token runs must cmp-equal the
#    committed per-token refs for ALL 5 prompts (config A = those files, re-hashed).
# 1. B = --kv-type f32 --flash-attn on, C = --kv-type f32 --flash-attn off, per-token, n=1 each, -v so llama's own
#    context-creation log proves the config engaged; C n=2 on orig.
# 2. C sub-layer dumps, the layerwise §5 regex/positions (p4 pos 0-3, orig pos 4,28); their logits must cmp C.
# 3. apr ON (APR_EMULATE_GGML_VECDOT=1) orig/p4 with the lambda-built binary e56bacdb, run on intel.
# intel, -t 8 / RAYON_NUM_THREADS=8, one job at a time, timeout + </dev/null on every binary, builds nice -n 10.
set -uo pipefail
L="${LLAMA_DIR:-$HOME/src/llama.cpp-d1d3c3396}"
K="$HOME/parity-ref/kvconfig-3091"; T=/tmp/kv3091
BIN="$L/apr-raw-logits/apr_raw_logits"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"
APRBIN="$T/qwen35_layer_obs"
RE='model\.input_embed|(attn_norm|conv_output_silu|q_conv_predelta|k_conv_predelta|v_conv_predelta|a_softplus|gate|beta_sigmoid|z|state_predelta|attn_output|new_state|final_output|linear_attn_out|attn_pregate|attn_gated|attn_residual|attn_post_norm|ffn_out|l_out)-[0-9]+|result_norm|result_output'
declare -A TXT=([orig]=/tmp/ref3303/prompt.txt [p1]=/tmp/flipvar3091/prompt-1.txt [p2]=/tmp/flipvar3091/prompt-2.txt [p3]=/tmp/flipvar3091/prompt-3.txt [p4]=/tmp/layer3091/prompt-4.txt)
declare -A IDS=([orig]="$T/ids/prompt_token_ids.txt" [p1]="$T/ids/prompt-1.ids" [p2]="$T/ids/prompt-2.ids" [p3]="$T/ids/prompt-3.ids" [p4]="$T/ids/prompt-4.ids")
declare -A REFA=([orig]="$HOME/parity-ref/qwen35-0.8b-q4km-d1d3c3396-per-token/per-token-run1.bin" [p1]="$HOME/parity-ref/variation-3091/p1-per-token.bin" [p2]="$HOME/parity-ref/variation-3091/p2-per-token.bin" [p3]="$HOME/parity-ref/variation-3091/p3-per-token.bin" [p4]="$HOME/parity-ref/variation-3091/p4-per-token.bin")
PROMPTS=(orig p1 p2 p3 p4)
mkdir -p "$K/runs"
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) host=$(hostname) nproc=$(nproc) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) free=$(df -h --output=avail "$HOME" | tail -1)"
echo "old producer sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
sha256sum "$MODEL" "$K/producer/apr_raw_logits.cpp" "$K/producer/build.sh" "$APRBIN" "${TXT[@]}" "${IDS[@]}" "${REFA[@]}"

echo "--- 0. build"
nice -n 10 bash "$K/producer/build.sh" > "$K/build.log" 2>&1; echo "build rc=$?"; cat "$K/build.log"

nids() { tr ',' '\n' < "$1" | grep -c .; }
run() { # name prompt extra...
  local name="$1" p="$2"; shift 2
  local n; n=$(nids "${IDS[$p]}")
  echo "run $name start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  timeout 900 "$BIN" -m "$MODEL" -f "${TXT[$p]}" -ngl 0 -t 8 -c "$n" -b "$n" --per-token "$@" \
    --raw-out "$K/runs/$name.bin" < /dev/null > "$K/runs/$name.log" 2>&1
  local rc=$?
  echo "run $name rc=$rc end_utc=$(date -u +%H:%M:%SZ)"
  if [ "$(sed -n 's/^apr_raw_logits: token_ids=//p' "$K/runs/$name.log")" = "$(tr -d '\n ' < "${IDS[$p]}")" ]; then echo "  ids == ${IDS[$p]##*/}"; else echo "  ids DIFFER"; fi
  grep -E '^apr_raw_logits: (kvconfig|dump)|type_k|flash_attn|K \(f|repack tensor blk.0.attn_q' "$K/runs/$name.log" | sed 's/^/  log| /'
  sha256sum "$K/runs/$name.bin" | sed 's/^/  /'
}

echo "--- 0b. config A: no-flag regression (rebuilt producer) vs committed per-token refs"
for p in "${PROMPTS[@]}"; do run "A-$p" "$p"; cmp "$K/runs/A-$p.bin" "${REFA[$p]}"; echo "A-$p (no-flag rebuilt) vs committed ${REFA[$p]##*/} cmp rc=$?"; done
echo "--- 0c. no-flag -v probe (orig): what llama logs for the default config"
run "A-v-orig" orig -v; cmp "$K/runs/A-v-orig.bin" "${REFA[orig]}"; echo "A-v-orig vs committed cmp rc=$?"

echo "--- 1. configs B (f32, FA on) and C (f32, FA off)"
for p in "${PROMPTS[@]}"; do run "B-$p" "$p" -v --kv-type f32 --flash-attn on; done
for p in "${PROMPTS[@]}"; do run "C-$p" "$p" -v --kv-type f32 --flash-attn off; done
run "C2-orig" orig -v --kv-type f32 --flash-attn off; cmp "$K/runs/C2-orig.bin" "$K/runs/C-orig.bin"; echo "C2-orig vs C-orig cmp rc=$?"
for p in "${PROMPTS[@]}"; do cmp -s "$K/runs/B-$p.bin" "${REFA[$p]}"; echo "B-$p vs A cmp rc=$?"; cmp -s "$K/runs/C-$p.bin" "${REFA[$p]}"; echo "C-$p vs A cmp rc=$?"; cmp -s "$K/runs/C-$p.bin" "$K/runs/B-$p.bin"; echo "C-$p vs B cmp rc=$?"; done

echo "--- 2. config C sub-layer dumps (free=$(df -h --output=avail "$HOME" | tail -1))"
rm -rf "${K:?}/runs/C-sub-p4" "${K:?}/runs/C-sub-orig"
run C-sub-p4 p4 --kv-type f32 --flash-attn off --dump-tensors "$K/runs/C-sub-p4" --dump-positions 0,1,2,3 --dump-regex "$RE"
cmp "$K/runs/C-sub-p4.bin" "$K/runs/C-p4.bin"; echo "C-sub-p4 logits vs C-p4 cmp rc=$?"
run C-sub-orig orig --kv-type f32 --flash-attn off --dump-tensors "$K/runs/C-sub-orig" --dump-positions 4,28 --dump-regex "$RE"
cmp "$K/runs/C-sub-orig.bin" "$K/runs/C-orig.bin"; echo "C-sub-orig logits vs C-orig cmp rc=$?"
for d in C-sub-p4 C-sub-orig; do
  ( cd "$K/runs/$d" && { head -1 manifest.tsv | tr -d '\n'; printf '\tsha256\n'; tail -n +2 manifest.tsv | while IFS=$'\t' read -r nm il pos shp nb f; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$nm" "$il" "$pos" "$shp" "$nb" "$f" "$(sha256sum "$f" | cut -d' ' -f1)"; done; } > manifest.sha256.tsv )
  echo "$d manifest rows=$(($(wc -l < "$K/runs/$d/manifest.sha256.tsv") - 1))"; sha256sum "$K/runs/$d/manifest.sha256.tsv"
done

echo "--- 3. apr ON on intel (lambda-built binary), RAYON_NUM_THREADS=8"
for p in orig p4; do
  echo "apr on-$p start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  env APR_EMULATE_GGML_VECDOT=1 RAYON_NUM_THREADS=8 timeout 1800 "$APRBIN" noop "$MODEL" "${IDS[$p]}" "$K/runs/apr-on-$p-intel.bin" < /dev/null > "$K/runs/apr-on-$p-intel.log" 2>&1
  echo "apr on-$p rc=$? end_utc=$(date -u +%H:%M:%SZ)"; grep -E '^qwen35_layer_obs:' "$K/runs/apr-on-$p-intel.log" | sed 's/^/  /'
  sha256sum "$K/runs/apr-on-$p-intel.bin" | sed 's/^/  /'
done
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
