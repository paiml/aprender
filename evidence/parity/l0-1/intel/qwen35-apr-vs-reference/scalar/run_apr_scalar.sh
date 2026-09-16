#!/usr/bin/env bash
# PMAT-3091 scalar (lambda): apr subjects with APR_EMULATE_GGML_VECDOT=scalar, the switch-OFF invariance, then the committed
# comparators (compare_raw_logits.py, compare_layerwise.py, layer_steps.py) against the SCALAR llama config-C refs.
set -uo pipefail
BIN="${BIN:-/mnt/nvme-raid0/targets/obs-3091/release/examples/qwen35_layer_obs}"
EV="${EV:-/mnt/nvme-raid0/agent-wt/layer-3091/evidence/parity/l0-1/intel/qwen35-apr-vs-reference}"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"; GGUF_PY="$HOME/src/llama.cpp-d1d3c3396/gguf-py"; EMBD=/tmp/layer3091/apr
W=/tmp/scalar3091; R="$W/ref"; O="$W/cmp"; LW="$EV/layerwise"; ENVV=APR_EMULATE_GGML_VECDOT
declare -A IDS=([orig]="$EV/prompt_token_ids.txt" [p1]="$EV/variation/prompt-1.ids" [p2]="$EV/variation/prompt-2.ids" [p3]="$EV/variation/prompt-3.ids" [p4]="$EV/variation/prompt-4.ids")
declare -A SUBJECT=([orig]=f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252 [p4]=92f1b54d504a53f758a56911c4779ce1900207bd89c78d58597e558ebc6d3ee9)
mkdir -p "$W/runs" "$O"
echo "obs_code=$(git -C /mnt/nvme-raid0/agent-wt/obs-3091 rev-parse HEAD) dirty_tracked=$(git -C /mnt/nvme-raid0/agent-wt/obs-3091 status --porcelain --untracked-files=no | wc -l) host=$(hostname) nproc=$(nproc) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
sha256sum "$BIN" "$MODEL" "${IDS[@]}" "$EV/compare_raw_logits.py" "$LW/compare_layerwise.py" "$LW/layer_steps.py" "$LW/layer_types.tsv"
run() { # label switch mode ids out-args...
  local label="$1" sw="$2"; shift 2
  local out="$W/runs/$label.bin"
  echo "run $label switch=$sw start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  if [ "$sw" = "-" ]; then env -u "$ENVV" timeout 1800 "$BIN" "$1" "$MODEL" "$2" "$out" "${@:3}" < /dev/null > "$W/runs/$label.log" 2>&1
  else env "$ENVV=$sw" timeout 1800 "$BIN" "$1" "$MODEL" "$2" "$out" "${@:3}" < /dev/null > "$W/runs/$label.log" 2>&1; fi
  local rc=$?
  echo "run $label rc=$rc end_utc=$(date -u +%H:%M:%SZ)"; grep -E '^qwen35_layer_obs:' "$W/runs/$label.log" | sed 's/^/  /'; sha256sum "$out" 2>/dev/null | sed 's/^/  /'
}
echo "--- 1. invariance (switch OFF)"
for p in orig p4; do run "off-$p" - noop "${IDS[$p]}"; got=$(sha256sum "$W/runs/off-$p.bin" | cut -d' ' -f1); [ "$got" = "${SUBJECT[$p]}" ] && echo "invariance $p sha_equal=true" || echo "invariance $p sha_equal=false got=$got"; done
echo "--- 2. scalar subjects"
for p in orig p4 p1 p2 p3; do run "scalar-$p" scalar noop "${IDS[$p]}"; done
run scalar2-orig scalar noop "${IDS[orig]}"; cmp "$W/runs/scalar2-orig.bin" "$W/runs/scalar-orig.bin"; echo "scalar2-orig vs scalar-orig cmp rc=$?"
for p in orig p4; do cmp -s "$W/runs/scalar-$p.bin" /tmp/emul3091/runs/on-$p.bin; echo "scalar-$p vs =1 on-$p cmp rc=$? (1 = scalar differs from native emulation)"; done
rm -rf "${W:?}/runs/scalar-dump-p4" "${W:?}/runs/scalar-dump-orig"
run scalar-dump-p4 scalar dump "${IDS[p4]}" 0,1,2,3 "$W/runs/scalar-dump-p4"
run scalar-dump-orig scalar dump "${IDS[orig]}" 4,28 "$W/runs/scalar-dump-orig"
cmp "$W/runs/scalar-dump-p4.bin" "$W/runs/scalar-p4.bin"; echo "scalar-dump-p4 logits vs scalar-p4 cmp rc=$?"
cmp "$W/runs/scalar-dump-orig.bin" "$W/runs/scalar-orig.bin"; echo "scalar-dump-orig logits vs scalar-orig cmp rc=$?"
echo "--- 3. compare vs llama SCALAR config C"
sha256sum "$R"/SC-*.bin
for p in orig p4 p1 p2 p3; do for m in scalar off; do
  timeout 600 python3 "$EV/compare_raw_logits.py" "$R/SC-$p.bin" "$W/runs/$m-$p.bin" --json "$O/logits-$p-$m-vs-SC.json" < /dev/null > "$O/logits-$p-$m-vs-SC.tsv" 2>&1; echo "compare_raw_logits $p $m vs SC rc=$?"
  [ -f "$W/runs/$m-$p.bin" ] || break
done; done
timeout 900 python3 "$LW/compare_layerwise.py" p4 "$R/SC-sub-p4" "$R/SC-p4.bin" "$EMBD/embd-p4" "$W/runs/scalar-dump-p4.bin" 0,1,2,3 "$LW/layer_types.tsv" "$MODEL" "$GGUF_PY" "$W/runs/scalar-dump-p4" < /dev/null > "$O/sublayer_p4_scalar_vs_SC.tsv" 2> "$O/sublayer_p4_scalar_vs_SC.selfcheck.txt"; echo "compare_layerwise p4 rc=$?"
timeout 900 python3 "$LW/compare_layerwise.py" orig "$R/SC-sub-orig" "$R/SC-orig.bin" "$EMBD/embd-p0" "$W/runs/scalar-dump-orig.bin" 4,28 "$LW/layer_types.tsv" "$MODEL" "$GGUF_PY" "$W/runs/scalar-dump-orig" < /dev/null > "$O/sublayer_orig_scalar_vs_SC.tsv" 2> "$O/sublayer_orig_scalar_vs_SC.selfcheck.txt"; echo "compare_layerwise orig rc=$?"
timeout 600 python3 "$LW/layer_steps.py" --ref p4:1 "$O/sublayer_p4_scalar_vs_SC.tsv" "$O/sublayer_orig_scalar_vs_SC.tsv" < /dev/null > "$O/layer_steps_scalar_vs_SC.tsv" 2>&1; echo "layer_steps rc=$?"
sha256sum "$O"/*
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
