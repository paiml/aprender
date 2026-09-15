#!/usr/bin/env bash
# PMAT-3091 scalar reference (intel): the producer linked against the SCALAR llama.cpp d1d3c3396 build, config C
# (--kv-type f32 --flash-attn off), per-token. -v on every logits run so llama's own log shows system_info / repack.
# Order = brief priority: orig (+n=2), p4, sub-layer dumps p4 pos 0-3 and orig pos 4,28 (logits cmp their runs), then p1-p3.
# -t 8, one job at a time, timeout + </dev/null on every binary.
set -uo pipefail
L="$HOME/src/llama.cpp-d1d3c3396"; S="$HOME/parity-ref/scalar-3091"; KV="$HOME/parity-ref/kvconfig-3091/runs"
BIN="$S/bin/apr_raw_logits"; MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"; I=/tmp/kv3091/ids
RE='model\.input_embed|(attn_norm|conv_output_silu|q_conv_predelta|k_conv_predelta|v_conv_predelta|a_softplus|gate|beta_sigmoid|z|state_predelta|attn_output|new_state|final_output|linear_attn_out|attn_pregate|attn_gated|attn_residual|attn_post_norm|ffn_out|l_out)-[0-9]+|result_norm|result_output'
declare -A TXT=([orig]=/tmp/ref3303/prompt.txt [p1]=/tmp/flipvar3091/prompt-1.txt [p2]=/tmp/flipvar3091/prompt-2.txt [p3]=/tmp/flipvar3091/prompt-3.txt [p4]=/tmp/layer3091/prompt-4.txt)
declare -A IDS=([orig]="$I/prompt_token_ids.txt" [p1]="$I/prompt-1.ids" [p2]="$I/prompt-2.ids" [p3]="$I/prompt-3.ids" [p4]="$I/prompt-4.ids")
mkdir -p "$S/runs"
echo "tree llama.cpp=$(git -C "$L" rev-parse --short=9 HEAD) host=$(hostname) nproc=$(nproc) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) free=$(df -h --output=avail "$HOME" | tail -1)"
sha256sum "$MODEL" "$BIN" "$S/producer/apr_raw_logits.cpp" "${TXT[@]}" "${IDS[@]}"
ldd "$BIN" | grep -E 'ggml|llama'
nids() { tr ',' '\n' < "$1" | grep -c .; }
run() { # name prompt extra...
  local name="$1" p="$2"; shift 2
  local n; n=$(nids "${IDS[$p]}")
  echo "run $name start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  timeout 5400 "$BIN" -m "$MODEL" -f "${TXT[$p]}" -ngl 0 -t 8 -c "$n" -b "$n" --per-token --kv-type f32 --flash-attn off "$@" \
    --raw-out "$S/runs/$name.bin" < /dev/null > "$S/runs/$name.log" 2>&1
  local rc=$?
  echo "run $name rc=$rc end_utc=$(date -u +%H:%M:%SZ)"
  if [ "$(sed -n 's/^apr_raw_logits: token_ids=//p' "$S/runs/$name.log")" = "$(tr -d '\n ' < "${IDS[$p]}")" ]; then echo "  ids == ${IDS[$p]##*/}"; else echo "  ids DIFFER"; fi
  echo "  repack_lines=$(grep -ci repack "$S/runs/$name.log") llamafile_lines=$(grep -ci llamafile "$S/runs/$name.log")"
  grep -E 'system_info|^apr_raw_logits: kvconfig|flash_attn|K \(f|CPU_REPACK|CPU model buffer' "$S/runs/$name.log" | sed 's/^/  log| /'
  sha256sum "$S/runs/$name.bin" | sed 's/^/  /'
}
manifest() {
  ( cd "$S/runs/$1" && { head -1 manifest.tsv | tr -d '\n'; printf '\tsha256\n'; tail -n +2 manifest.tsv | while IFS=$'\t' read -r nm il pos shp nb f; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$nm" "$il" "$pos" "$shp" "$nb" "$f" "$(sha256sum "$f" | cut -d' ' -f1)"; done; } > manifest.sha256.tsv )
  echo "$1 manifest rows=$(($(wc -l < "$S/runs/$1/manifest.sha256.tsv") - 1))"; sha256sum "$S/runs/$1/manifest.sha256.tsv"
}
echo "--- 1. scalar C, orig n=2, p4"
run SC-orig orig -v
run SC2-orig orig -v; cmp "$S/runs/SC2-orig.bin" "$S/runs/SC-orig.bin"; echo "SC2-orig vs SC-orig cmp rc=$?"
run SC-p4 p4 -v
for p in orig p4; do cmp -s "$S/runs/SC-$p.bin" "$KV/C-$p.bin"; echo "SC-$p vs native-build C-$p cmp rc=$? (1 = the scalar build changed the arithmetic)"; done
echo "--- 2. scalar C sub-layer dumps"
rm -rf "$S/runs/SC-sub-p4" "$S/runs/SC-sub-orig"
run SC-sub-p4 p4 --dump-tensors "$S/runs/SC-sub-p4" --dump-positions 0,1,2,3 --dump-regex "$RE"
cmp "$S/runs/SC-sub-p4.bin" "$S/runs/SC-p4.bin"; echo "SC-sub-p4 logits vs SC-p4 cmp rc=$?"; manifest SC-sub-p4
run SC-sub-orig orig --dump-tensors "$S/runs/SC-sub-orig" --dump-positions 4,28 --dump-regex "$RE"
cmp "$S/runs/SC-sub-orig.bin" "$S/runs/SC-orig.bin"; echo "SC-sub-orig logits vs SC-orig cmp rc=$?"; manifest SC-sub-orig
echo "--- 3. p1-p3 (budget-permitting)"
for p in p1 p2 p3; do run "SC-$p" "$p" -v; done
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
