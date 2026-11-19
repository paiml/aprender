#!/usr/bin/env bash
# LAB lane (#6570): apr GPU decode tok/s vs pinned llama.cpp, same GGUF, same GPU.
# Writes a ratio receipt. Has NO threshold and never fails on the ratio: a LAB
# check cannot block (GATE/LAB rule). Exit 0 with status=NOT_MEASURED when a
# side is absent, so a red is a ticket and not a stop.
# Usage: APR_BIN=... APR_GGUF=... LLAMA_BENCH_PATH=... OUT=receipt.json bash scripts/llamacpp_decode_ratio_lab.sh
set -uo pipefail
: "${APR_BIN:?APR_BIN}"; : "${APR_GGUF:?APR_GGUF}"; : "${OUT:?OUT}"
PROMPT="${PROMPT:-Write a short story about a robot.}"
TRIALS="${TRIALS:-7}"; LO=128; HI=384
status=NOT_MEASURED; reason=""; apr_tps=""; llama_tps=""; ratio=""

# One apr run -> "<tokens> <ms>" from the 'Generated N tokens in Xms' line.
apr_once() {
  local txt rc
  txt=$("$APR_BIN" run "$APR_GGUF" --prompt "$PROMPT" --max-tokens "$1" --gpu --benchmark 2>&1); rc=$?
  [ "$rc" -eq 0 ] || return 1
  printf '%s\n' "$txt" | awk '/Generated/ && /tokens in/ { for (i=1;i<=NF;i++) if ($i=="in") { ms=$(i+1); sub(/ms.*/,"",ms) } print ms; exit }'
}
median() { sort -g | awk '{a[NR]=$1} END{ if(NR) print a[int((NR+1)/2)] }'; }

measure() {
  [ -f "$APR_BIN" ] && [ -x "$APR_BIN" ] || { reason="apr binary missing"; return; }
  [ -f "$APR_GGUF" ] || { reason="gguf missing"; return; }
  . scripts/llama_bin.sh || { reason="llama_bin.sh not sourceable"; return; }
  llama_bin_resolve; local prc=$?
  [ "$prc" -eq 0 ] || { reason="llama.cpp comparator unpinned rc=$prc ${LLAMA_PIN_REASON:-}"; return; }
  local i lo hi diffs="" v
  for i in $(seq "$TRIALS"); do
    lo=$(apr_once "$LO") || continue; hi=$(apr_once "$HI") || continue
    [ -n "$lo" ] && [ -n "$hi" ] || continue
    v=$(awk -v l="$lo" -v h="$hi" -v a="$LO" -v b="$HI" 'BEGIN{ if (h>l) printf "%.2f", (b-a)/(h-l)*1000 }')
    [ -n "$v" ] && diffs="$diffs$v"$'\n'
  done
  apr_tps=$(printf '%s' "$diffs" | median)
  [ -n "$apr_tps" ] || { reason="apr decode not measurable (CPU fallback or run failed)"; return; }
  local csv; csv=$("$LLAMA_BENCH" -m "$APR_GGUF" -p 0 -n "$HI" -ngl 99 -r 5 -o csv 2>/dev/null); prc=$?
  [ "$prc" -eq 0 ] || { reason="llama-bench rc=$prc"; return; }
  llama_tps=$(printf '%s\n' "$csv" | tail -n 1 | awk -F, '{ v=$(NF-1); gsub(/"/,"",v); print v }')
  [ -n "$llama_tps" ] || { reason="llama-bench produced no tg row"; return; }
  ratio=$(awk -v a="$apr_tps" -v l="$llama_tps" 'BEGIN{ if (l>0) printf "%.4f", a/l }')
  [ -n "$ratio" ] && status=MEASURED
}
measure
mkdir -p "$(dirname "$OUT")"
printf '{"lane":"llamacpp-decode-ratio-lab","status":"%s","reason":"%s","apr_decode_tps_median":"%s","llamacpp_tg_tps":"%s","ratio_apr_over_llamacpp":"%s","trials":%s,"llama_build":"%s","gguf":"%s","threshold":null}\n' \
  "$status" "$reason" "$apr_tps" "$llama_tps" "$ratio" "$TRIALS" "${LLAMA_BUILD:-}" "$(basename "$APR_GGUF")" >"$OUT"
cat "$OUT"
exit 0
